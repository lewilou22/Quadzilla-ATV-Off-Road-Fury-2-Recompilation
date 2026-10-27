// ATV Offroad Fury 2 (SCUS_972.11) online play: network HLE.
//
// The retail game talks to its IOP network stack (INET/LIBNETB over MSIFRPC, plus SCEA's
// ezNetCnf/ezNetCtl helpers) and nothing on our IOP side implements those modules. Instead of
// emulating the msifrpc protocol and libnetb's DMA ring buffers, the EE-side seams are replaced:
//
//   * ezNetCnf / ezNetCtl clients: one synthetic "Your Network Configuration" entry, connect always
//     succeeds with the link up (the host OS owns the real network).
//   * SCE-RT NetLib's PS2 platform layer (the socket API Medius/MGCL/DME sit on): host sockets.
//   * PSXNetwork's DNS helper: host resolver.
//
// Name lookups go to a configurable DNS server, defaulting to PS Rewired, so the game's hardcoded
// SCEA hostnames (atv2-prod2.pdonline.scea.com) resolve to the revival servers. There is no DNAS in
// this game. Environment:
//   ATV2_NET_DNS=<ipv4>          DNS server for game lookups (default 67.222.156.250, PS Rewired)
//   ATV2_NET_HOSTS=name=ip;...   static overrides, checked before DNS
//   ATV2_NET_CONNECT_MS=<ms>     TCP connect timeout (default 8000)
//   ATV2_NET_TRACE=1             log every socket call
//   ATV2_NET_HLE=0               disable this module (game sees no network stack)

// Runtime headers first: windows.h defines macros (EXCEPTION_BREAKPOINT, ...) that collide with them.
#include "ps2_runtime.h"
#include "ps2_runtime_macros.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define NOGDI  // raylib: Rectangle
#define NOUSER // raylib: CloseWindow, ShowCursor
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#ifndef SIO_UDP_CONNRESET
#define SIO_UDP_CONNRESET _WSAIOW(IOC_VENDOR, 12)
#endif
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <algorithm>
#include <string>

void atv2ArmWriteWatch(uint8_t *rdram, uint32_t guestAddr, uint32_t len); // Kernel/Atv2WriteWatch.cpp
void atv2DumpWriteWatchRing();

namespace
{
#ifdef _WIN32
    using HostSocket = SOCKET;
    constexpr HostSocket kBadSocket = INVALID_SOCKET;
    int lastSockError() { return WSAGetLastError(); }
    bool wouldBlock(int e) { return e == WSAEWOULDBLOCK || e == WSAEINPROGRESS; }
    void closeHostSocket(HostSocket s) { closesocket(s); }
    bool setNonBlocking(HostSocket s)
    {
        u_long on = 1;
        return ioctlsocket(s, FIONBIO, &on) == 0;
    }
    int bytesPending(HostSocket s)
    {
        u_long n = 0;
        return ioctlsocket(s, FIONREAD, &n) == 0 ? static_cast<int>(n) : -1;
    }
#else
    using HostSocket = int;
    constexpr HostSocket kBadSocket = -1;
    int lastSockError() { return errno; }
    bool wouldBlock(int e) { return e == EWOULDBLOCK || e == EAGAIN || e == EINPROGRESS; }
    void closeHostSocket(HostSocket s) { close(s); }
    bool setNonBlocking(HostSocket s) { return fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) | O_NONBLOCK) == 0; }
    int bytesPending(HostSocket s)
    {
        int n = 0;
        return ioctl(s, FIONREAD, &n) == 0 ? n : -1;
    }
#endif

    // NetLib platform-layer status codes (as returned by the retail implementations).
    constexpr int32_t kNetOk = 0;
    constexpr int32_t kNetError = 1;
    constexpr int32_t kNetBadParam = 2;
    constexpr int32_t kNetUnsupported = 3;
    constexpr int32_t kNetAlreadyInit = 4;
    constexpr int32_t kNetNotInit = 5;
    constexpr int32_t kNetClosed = 0xD;

    // Game globals owned by the NetLib platform layer.
    constexpr uint32_t kNetInitFlagAddr = 0x004144B8u; // byte, 1 = initialized
    constexpr uint32_t kNetLocalIpAddr = 0x004144C0u;  // char[24], dotted local IP
    constexpr uint32_t kNetDnsIpAddr = 0x004144D8u;    // char[20], dotted DNS server
    constexpr uint32_t kNetIfIdAddr = 0x004144ECu;     // int, active interface id
    // ezNetCnf / ezNetCtl client "bound" flags.
    constexpr uint32_t kEzNetCnfBoundAddr = 0x0041D508u;
    constexpr uint32_t kEzNetCtlBoundAddr = 0x0041D510u;

    struct NetSocket
    {
        HostSocket s = kBadSocket;
        int type = 0; // NetLib type: 0 TCP client, 1 TCP listen, 2 UDP bound, 3 UDP auto port
    };

    std::mutex g_netMutex;
    std::map<int32_t, NetSocket> g_sockets;
    int32_t g_nextSocketId = 1;
    bool g_wsaStarted = false;
    std::string g_localIp = "0.0.0.0";

    bool netTrace()
    {
        static const bool s_trace = []() {
            const char *v = std::getenv("ATV2_NET_TRACE");
            return v != nullptr && v[0] != '0';
        }();
        return s_trace;
    }

    // " | aa bb cc ..." for trace lines (first 1024 bytes).
    std::string hexDump(const uint8_t *p, size_t n)
    {
        static const char kHex[] = "0123456789abcdef";
        std::string out = n ? " |" : "";
        for (size_t i = 0; i < n && i < 1024u; ++i)
        {
            out.push_back(' ');
            out.push_back(kHex[p[i] >> 4]);
            out.push_back(kHex[p[i] & 15]);
        }
        return out;
    }

    std::string dnsServer()
    {
        const char *v = std::getenv("ATV2_NET_DNS");
        return (v != nullptr && v[0] != '\0') ? std::string(v) : std::string("67.222.156.250");
    }

    uint8_t *guestPtr(uint8_t *rdram, uint32_t addr) { return rdram + (addr & PS2_RAM_MASK); }

    std::string readGuestString(uint8_t *rdram, uint32_t addr, size_t maxLen = 255)
    {
        std::string out;
        if (addr == 0u)
            return out;
        for (size_t i = 0; i < maxLen; ++i)
        {
            const char c = static_cast<char>(*guestPtr(rdram, addr + static_cast<uint32_t>(i)));
            if (c == '\0')
                break;
            out.push_back(c);
        }
        return out;
    }

    void writeGuestString(uint8_t *rdram, uint32_t addr, const std::string &s, size_t cap)
    {
        if (addr == 0u || cap == 0u)
            return;
        const size_t n = std::min(s.size(), cap - 1u);
        for (size_t i = 0; i < n; ++i)
            *guestPtr(rdram, addr + static_cast<uint32_t>(i)) = static_cast<uint8_t>(s[i]);
        *guestPtr(rdram, addr + static_cast<uint32_t>(n)) = 0u;
    }

    void writeGuest16(uint8_t *rdram, uint32_t addr, uint16_t v)
    {
        if (addr != 0u)
            std::memcpy(guestPtr(rdram, addr), &v, sizeof(v));
    }

    void writeGuest32(uint8_t *rdram, uint32_t addr, uint32_t v)
    {
        if (addr != 0u)
            std::memcpy(guestPtr(rdram, addr), &v, sizeof(v));
    }

    // ATV2_NET_TRACE crash aid: the guest context of the last HLE call, dumped on a host access
    // violation so a fault deep in a libc stub can be tied back to its guest caller.
    R5900Context *g_lastCtx = nullptr;
    uint8_t *g_lastRdram = nullptr;
#ifdef _WIN32
    LONG CALLBACK netCrashHandler(EXCEPTION_POINTERS *info)
    {
        if (info->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || g_lastCtx == nullptr)
            return EXCEPTION_CONTINUE_SEARCH;
        static bool s_dumped = false;
        if (s_dumped)
            return EXCEPTION_CONTINUE_SEARCH;
        s_dumped = true;
        const R5900Context *c = g_lastCtx;
        std::cerr << "[ATV2:NET] host AV at 0x" << std::hex << info->ExceptionRecord->ExceptionInformation[1]
                  << " guest pc=0x" << c->pc << " ra=0x" << getRegU32(c, 31) << " sp=0x" << getRegU32(c, 29)
                  << " a0=0x" << getRegU32(c, 4) << " a1=0x" << getRegU32(c, 5) << " a2=0x" << getRegU32(c, 6)
                  << " a3=0x" << getRegU32(c, 7) << std::dec << std::endl;
        const uint32_t sp = getRegU32(c, 29);
        int shown = 0;
        for (uint32_t a = sp & ~3u; a < sp + 0x800u && a < 0x02000000u && shown < 24; a += 4u)
        {
            uint32_t w = 0;
            std::memcpy(&w, g_lastRdram + (a & PS2_RAM_MASK), 4);
            if (w >= 0x00100008u && w < 0x00400000u && (w & 3u) == 0u)
            {
                uint32_t call = 0;
                std::memcpy(&call, g_lastRdram + ((w - 8u) & PS2_RAM_MASK), 4);
                if ((call >> 26) == 3u || (call & 0xFC00003Fu) == 0x00000009u)
                {
                    std::cerr << "[ATV2:NET]   stack @0x" << std::hex << a << " ret=0x" << w << std::dec << std::endl;
                    ++shown;
                }
            }
        }
        return EXCEPTION_CONTINUE_SEARCH;
    }
#endif

    void returnFromOverride(R5900Context *ctx, int32_t v0)
    {
        g_lastCtx = ctx;
        SET_GPR_S32(ctx, 2, v0);
        ctx->pc = getRegU32(ctx, 31);
    }

    bool ensureSockets()
    {
#ifdef _WIN32
        if (!g_wsaStarted)
        {
            WSADATA data{};
            if (WSAStartup(MAKEWORD(2, 2), &data) != 0)
                return false;
            g_wsaStarted = true;
        }
#endif
        return true;
    }

    std::string ipToString(const in_addr &a)
    {
        char buf[INET_ADDRSTRLEN] = {};
        inet_ntop(AF_INET, &a, buf, sizeof(buf));
        return buf;
    }

    // ---- DNS ---------------------------------------------------------------------------------

    bool lookupStaticHost(const std::string &name, in_addr &out)
    {
        const char *v = std::getenv("ATV2_NET_HOSTS");
        if (v == nullptr)
            return false;
        std::string all(v);
        size_t pos = 0;
        while (pos < all.size())
        {
            size_t end = all.find(';', pos);
            if (end == std::string::npos)
                end = all.size();
            const std::string entry = all.substr(pos, end - pos);
            const size_t eq = entry.find('=');
            if (eq != std::string::npos && entry.compare(0, eq, name) == 0 &&
                inet_pton(AF_INET, entry.substr(eq + 1).c_str(), &out) == 1)
                return true;
            pos = end + 1;
        }
        return false;
    }

    // Minimal A-record query against the configured server (the host's own resolver doesn't know
    // the revival servers' answers for the old SCEA names).
    bool dnsQueryA(const std::string &name, const std::string &server, in_addr &out)
    {
        sockaddr_in srv{};
        srv.sin_family = AF_INET;
        srv.sin_port = htons(53);
        if (inet_pton(AF_INET, server.c_str(), &srv.sin_addr) != 1)
            return false;

        static std::mt19937 s_rng(static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
        const uint16_t id = static_cast<uint16_t>(s_rng());
        uint8_t q[512] = {};
        size_t len = 0;
        q[len++] = static_cast<uint8_t>(id >> 8);
        q[len++] = static_cast<uint8_t>(id);
        q[len++] = 0x01; // RD
        q[len++] = 0x00;
        q[len++] = 0x00;
        q[len++] = 0x01; // QDCOUNT
        len += 6;
        size_t labelStart = 0;
        for (size_t i = 0; i <= name.size(); ++i)
        {
            if (i == name.size() || name[i] == '.')
            {
                const size_t labelLen = i - labelStart;
                if (labelLen == 0 || labelLen > 63 || len + labelLen + 6 >= sizeof(q))
                    return false;
                q[len++] = static_cast<uint8_t>(labelLen);
                std::memcpy(q + len, name.data() + labelStart, labelLen);
                len += labelLen;
                labelStart = i + 1;
            }
        }
        q[len++] = 0;
        q[len++] = 0x00;
        q[len++] = 0x01; // A
        q[len++] = 0x00;
        q[len++] = 0x01; // IN

        HostSocket s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (s == kBadSocket)
            return false;
        bool ok = false;
        for (int attempt = 0; attempt < 3 && !ok; ++attempt)
        {
            if (sendto(s, reinterpret_cast<const char *>(q), static_cast<int>(len), 0,
                       reinterpret_cast<const sockaddr *>(&srv), sizeof(srv)) < 0)
                break;
            fd_set rd;
            FD_ZERO(&rd);
            FD_SET(s, &rd);
            timeval tv{1, 500000};
            if (select(static_cast<int>(s) + 1, &rd, nullptr, nullptr, &tv) <= 0)
                continue;
            uint8_t r[1500];
            const int n = recv(s, reinterpret_cast<char *>(r), sizeof(r), 0);
            if (n < 12 || r[0] != q[0] || r[1] != q[1] || (r[3] & 0x0F) != 0)
                break; // short reply, wrong id, or NXDOMAIN/SERVFAIL
            const int qd = (r[4] << 8) | r[5];
            const int an = (r[6] << 8) | r[7];
            size_t p = 12;
            auto skipName = [&](size_t at) -> size_t {
                while (at < static_cast<size_t>(n))
                {
                    const uint8_t l = r[at];
                    if (l == 0)
                        return at + 1;
                    if ((l & 0xC0) == 0xC0)
                        return at + 2;
                    at += 1u + l;
                }
                return static_cast<size_t>(n);
            };
            for (int i = 0; i < qd; ++i)
                p = skipName(p) + 4;
            for (int i = 0; i < an && p + 10 <= static_cast<size_t>(n); ++i)
            {
                p = skipName(p);
                if (p + 10 > static_cast<size_t>(n))
                    break;
                const int type = (r[p] << 8) | r[p + 1];
                const int rdlen = (r[p + 8] << 8) | r[p + 9];
                p += 10;
                if (type == 1 && rdlen == 4 && p + 4 <= static_cast<size_t>(n))
                {
                    std::memcpy(&out, r + p, 4);
                    ok = true;
                    break;
                }
                p += static_cast<size_t>(rdlen);
            }
            break;
        }
        closeHostSocket(s);
        return ok;
    }

    bool resolveHost(const std::string &name, in_addr &out)
    {
        if (name.empty())
            return false;
        if (inet_pton(AF_INET, name.c_str(), &out) == 1)
            return true;
        if (lookupStaticHost(name, out))
            return true;
        if (!ensureSockets())
            return false;
        if (dnsQueryA(name, dnsServer(), out))
            return true;
        addrinfo hints{};
        hints.ai_family = AF_INET;
        addrinfo *res = nullptr;
        if (getaddrinfo(name.c_str(), nullptr, &hints, &res) == 0 && res != nullptr)
        {
            out = reinterpret_cast<sockaddr_in *>(res->ai_addr)->sin_addr;
            freeaddrinfo(res);
            return true;
        }
        return false;
    }

    bool resolveAndLog(const std::string &name, in_addr &out)
    {
        const bool ok = resolveHost(name, out);
        std::cerr << "[ATV2:NET] resolve '" << name << "' -> " << (ok ? ipToString(out) : std::string("FAILED"))
                  << " (dns " << dnsServer() << ")" << std::endl;
        return ok;
    }

    std::string detectLocalIp()
    {
        HostSocket s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (s == kBadSocket)
            return "0.0.0.0";
        sockaddr_in to{};
        to.sin_family = AF_INET;
        to.sin_port = htons(53);
        inet_pton(AF_INET, dnsServer().c_str(), &to.sin_addr);
        std::string ip = "0.0.0.0";
        if (connect(s, reinterpret_cast<const sockaddr *>(&to), sizeof(to)) == 0)
        {
            sockaddr_in me{};
            socklen_t meLen = sizeof(me);
            if (getsockname(s, reinterpret_cast<sockaddr *>(&me), &meLen) == 0)
                ip = ipToString(me.sin_addr);
        }
        closeHostSocket(s);
        return ip;
    }

    NetSocket *findSocket(int32_t id)
    {
        auto it = g_sockets.find(id);
        return it == g_sockets.end() ? nullptr : &it->second;
    }

    bool netInitialized(uint8_t *rdram) { return *guestPtr(rdram, kNetInitFlagAddr) == 1u; }

    // ---- ezNetCnf / ezNetCtl ---------------------------------------------------------------------

    constexpr uint32_t kEzListHeader = 0x30u;
    constexpr uint32_t kEzEntrySize = 0x68u;
    constexpr uint32_t kEzMaxEntries = 6u;
    constexpr uint32_t kNetcnfDataSize = 0x1340u;

    // void ezNetCnfBind(void) @ 0x003B1FC8
    void ezNetCnfBind(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        writeGuest32(rdram, kEzNetCnfBoundAddr, 1u);
        std::cerr << "[ATV2:NET] ezNetCnf bound (HLE)" << std::endl;
        returnFromOverride(ctx, 1);
    }

    // int ezNetCnfUnbind(void) @ 0x003B2090
    void ezNetCnfUnbind(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        writeGuest32(rdram, kEzNetCnfBoundAddr, 0u);
        returnFromOverride(ctx, 0);
    }

    // int ezNetCnfGetCount(const char *path, int type) @ 0x003B2118
    void ezNetCnfGetCount(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        std::cerr << "[ATV2:NET] ezNetCnfGetCount('" << readGuestString(rdram, getRegU32(ctx, 4)) << "') -> 1" << std::endl;
        returnFromOverride(ctx, 1);
    }

    // int ezNetCnfGetCombinationList(const char *path, ezNetCnfCombinationList *list) @ 0x003B21D8
    // List: int count, int defaultNumber, ..., entries of 0x68 bytes from +0x30:
    //   +0 valid, +4 type, +8 combination name, +0x28 UI line 1, +0x48 UI line 2.
    // Returns the number of entry slots (6) on success.
    void ezNetCnfGetCombinationList(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        const uint32_t list = getRegU32(ctx, 5);
        std::memset(guestPtr(rdram, list), 0, kEzListHeader + kEzEntrySize * kEzMaxEntries);
        writeGuest32(rdram, list + 0u, 1u);
        writeGuest32(rdram, list + 4u, 1u);
        const uint32_t e = list + kEzListHeader;
        writeGuest32(rdram, e + 0u, 1u);
        writeGuestString(rdram, e + 0x08u, "Combination1", 0x20u);
        writeGuestString(rdram, e + 0x28u, "Host PC network", 0x20u);
        writeGuestString(rdram, e + 0x48u, "Recomp network HLE", 0x20u);
        std::cerr << "[ATV2:NET] ezNetCnfGetCombinationList('" << readGuestString(rdram, getRegU32(ctx, 4))
                  << "') -> 1 synthetic entry" << std::endl;
        returnFromOverride(ctx, static_cast<int32_t>(kEzMaxEntries));
    }

    // int ezNetCnfSetDefault(const char *path, const char *combination) @ 0x003B2398
    void ezNetCnfSetDefault(uint8_t *, R5900Context *ctx, PS2Runtime *) { returnFromOverride(ctx, 0); }

    // int ezNetCnfGetNetcnfifData(const char *path, const char *combination, void *data) @ 0x003B2460
    // +0xE00 is the device vendor: the config page rejects anything but "SCE" (the official Network
    // Adaptor) as unsupported hardware, and PSXNetwork::Connect then takes the SMAP Ethernet path.
    // +0x130C = 2 / +0x1325 select PPPoE handling; leave them off.
    void ezNetCnfGetNetcnfifData(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        const uint32_t data = getRegU32(ctx, 6);
        std::memset(guestPtr(rdram, data), 0, kNetcnfDataSize);
        writeGuestString(rdram, data + 0xE00u, "SCE", 0x20u);
        writeGuest32(rdram, data + 0x130Cu, 1u);
        std::cerr << "[ATV2:NET] ezNetCnfGetNetcnfifData('" << readGuestString(rdram, getRegU32(ctx, 5)) << "')" << std::endl;
        returnFromOverride(ctx, 0);
    }

    // void ezNetCtlBind(void) @ 0x003B2538
    void ezNetCtlBind(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        writeGuest32(rdram, kEzNetCtlBoundAddr, 1u);
        std::cerr << "[ATV2:NET] ezNetCtl bound (HLE)" << std::endl;
        returnFromOverride(ctx, 1);
    }

    // int ezNetCtlUnbind(void) @ 0x003B2600
    void ezNetCtlUnbind(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        writeGuest32(rdram, kEzNetCtlBoundAddr, 0u);
        returnFromOverride(ctx, 0);
    }

    // int ezNetCtlSetProperties(const void *netcnfData) @ 0x003B2688; > 0 = accepted
    void ezNetCtlSetProperties(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        std::cerr << "[ATV2:NET] ezNetCtlSetProperties -> connecting via host network" << std::endl;
        returnFromOverride(ctx, 1);
    }

    // int ezNetCtlSetCombination(const char *path, const char *combination) @ 0x003B27A8
    void ezNetCtlSetCombination(uint8_t *, R5900Context *ctx, PS2Runtime *) { returnFromOverride(ctx, 0); }

    // int ezNetCtlGetStatus(ezNetCtlStatus *status) @ 0x003B2870
    // 0x110 bytes: +0 error, +8 physical link, +0xC state (3 = connected, 5 = failed), +0x10 message.
    void ezNetCtlGetStatus(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        const uint32_t st = getRegU32(ctx, 4);
        g_lastRdram = rdram;
        std::memset(guestPtr(rdram, st), 0, 0x110u);
        writeGuest32(rdram, st + 0x08u, 1u);
        writeGuest32(rdram, st + 0x0Cu, 3u);
        writeGuestString(rdram, st + 0x10u, "Connected", 0x100u);
        static bool s_logged = false;
        if (!s_logged)
        {
            s_logged = true;
            std::cerr << "[ATV2:NET] ezNetCtlGetStatus -> link up, state 3 (connected)" << std::endl;
        }
        // ATV2_NET_TRACE: PSXNetwork state (+0x50), failure reason (+0x54), mode (+0x60) and device
        // path (+0x64) decide the "Connection Failed!" alert; log them whenever they change.
        if (netTrace())
        {
            uint32_t game = 0, psx = 0;
            std::memcpy(&game, guestPtr(rdram, 0x00404818u), 4);
            if (game != 0u)
                std::memcpy(&psx, guestPtr(rdram, game + 0x2E4u), 4);
            static uint32_t s_last[4] = {~0u, ~0u, ~0u, ~0u};
            if (psx != 0u)
            {
                uint32_t v[4];
                std::memcpy(&v[0], guestPtr(rdram, psx + 0x50u), 4);
                std::memcpy(&v[1], guestPtr(rdram, psx + 0x54u), 4);
                std::memcpy(&v[2], guestPtr(rdram, psx + 0x60u), 4);
                std::memcpy(&v[3], guestPtr(rdram, psx + 0x64u), 4);
                if (std::memcmp(v, s_last, sizeof(v)) != 0)
                {
                    std::memcpy(s_last, v, sizeof(v));
                    std::cerr << "[ATV2:NET] PSXNetwork state=" << v[0] << " reason=" << v[1] << " mode=" << v[2]
                              << " dev=" << v[3] << " (GetStatus ra=0x" << std::hex << getRegU32(ctx, 31) << std::dec
                              << ")" << std::endl;
                }
            }
            // The online page's step machine (0x2A21A0) waits on two UI network managers
            // (*0x4B7E68 login, *0x4B7E70 lobby/MGCL): status byte +0x131 (1 ok, 3 failed) and the
            // Medius error at +0x2C4 choose the alert.
            static int32_t s_mgr[6] = {-99999, -99999, -99999, -99999, -99999, -99999};
            int32_t m[6] = {-1, 0, -1, 0, -1, 0};
            for (int i = 0; i < 2; ++i)
            {
                uint32_t obj = 0;
                std::memcpy(&obj, guestPtr(rdram, i == 0 ? 0x004B7E68u : 0x004B7E70u), 4);
                if (obj != 0u)
                {
                    m[i * 2] = *guestPtr(rdram, obj + 0x131u);
                    std::memcpy(&m[i * 2 + 1], guestPtr(rdram, obj + 0x2C4u), 4);
                    if (i == 1)
                    {
                        m[4] = *guestPtr(rdram, obj + 0x122u); // lobby step (0x3177C0)
                        std::memcpy(&m[5], guestPtr(rdram, obj + 0x104u), 4); // timeout flag
                    }
                }
            }
            if (std::memcmp(m, s_mgr, sizeof(m)) != 0)
            {
                std::memcpy(s_mgr, m, sizeof(m));
                std::cerr << "[ATV2:NET] UI net managers: login status=" << m[0] << " err=" << m[1]
                          << " | lobby status=" << m[2] << " err=" << m[3] << " step=" << m[4] << " timeout=" << m[5]
                          << std::endl;
            }
        }
        returnFromOverride(ctx, 0);
    }

    // ezNetCtl commands 5 and 6 (one int argument, no reply) @ 0x003B2988 / 0x003B2A00
    void ezNetCtlCommand(uint8_t *, R5900Context *ctx, PS2Runtime *) { returnFromOverride(ctx, 0); }

    // ---- NetLib PS2 platform layer -----------------------------------------------------------------

    // int NetInit(void) @ 0x0037E120
    void netInit(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        std::lock_guard<std::mutex> lock(g_netMutex);
        if (netInitialized(rdram))
        {
            returnFromOverride(ctx, kNetAlreadyInit);
            return;
        }
        if (!ensureSockets())
        {
            std::cerr << "[ATV2:NET] NetInit: socket layer unavailable" << std::endl;
            returnFromOverride(ctx, kNetError);
            return;
        }
        g_localIp = detectLocalIp();
        writeGuestString(rdram, kNetLocalIpAddr, g_localIp, 0x18u);
        writeGuestString(rdram, kNetDnsIpAddr, dnsServer(), 0x14u);
        writeGuest32(rdram, kNetIfIdAddr, 1u);
        *guestPtr(rdram, kNetInitFlagAddr) = 1u;
        std::cerr << "[ATV2:NET] NetInit: local " << g_localIp << ", dns " << dnsServer() << std::endl;
        returnFromOverride(ctx, kNetOk);
    }

    // int NetShutdown(void) @ 0x0037E1E8
    void netShutdown(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        std::lock_guard<std::mutex> lock(g_netMutex);
        if (!netInitialized(rdram))
        {
            returnFromOverride(ctx, kNetNotInit);
            return;
        }
        for (auto &kv : g_sockets)
            closeHostSocket(kv.second.s);
        g_sockets.clear();
        *guestPtr(rdram, kNetInitFlagAddr) = 0u;
        writeGuest32(rdram, kNetIfIdAddr, 0xFFFFFFFFu);
        std::cerr << "[ATV2:NET] NetShutdown" << std::endl;
        returnFromOverride(ctx, kNetOk);
    }

    bool connectWithTimeout(HostSocket s, const sockaddr_in &to)
    {
        static const int s_timeoutMs = []() {
            const char *v = std::getenv("ATV2_NET_CONNECT_MS");
            return v != nullptr ? std::max(100, std::atoi(v)) : 8000;
        }();
        if (connect(s, reinterpret_cast<const sockaddr *>(&to), sizeof(to)) == 0)
            return true;
        if (!wouldBlock(lastSockError()))
            return false;
        fd_set wr, ex;
        FD_ZERO(&wr);
        FD_ZERO(&ex);
        FD_SET(s, &wr);
        FD_SET(s, &ex);
        timeval tv{s_timeoutMs / 1000, (s_timeoutMs % 1000) * 1000};
        if (select(static_cast<int>(s) + 1, nullptr, &wr, &ex, &tv) <= 0 || FD_ISSET(s, &ex))
            return false;
        int err = 0;
        socklen_t errLen = sizeof(err);
        getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char *>(&err), &errLen);
        return err == 0;
    }

    // int NetOpen(int *sockOut, const char *host, u16 port, const NetOpenParams *params) @ 0x0037E268
    // params->type: 0 TCP connect to host:port; 1, 2 and 3 are datagram sockets (retail maps 1 to
    // libinet's packet-ring mode with per-record peer addresses, which DME reads with RecvFrom):
    // 1 bound to port (params[1] must be >= 1), 2 bound to port, 3 on an automatic port.
    void netOpen(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        const uint32_t outAddr = getRegU32(ctx, 4);
        const std::string host = readGuestString(rdram, getRegU32(ctx, 5));
        const uint16_t port = static_cast<uint16_t>(getRegU32(ctx, 6));
        const uint32_t params = getRegU32(ctx, 7);
        std::lock_guard<std::mutex> lock(g_netMutex);
        if (outAddr == 0u)
        {
            returnFromOverride(ctx, kNetBadParam);
            return;
        }
        writeGuest32(rdram, outAddr, 0u);
        if (!netInitialized(rdram))
        {
            returnFromOverride(ctx, kNetNotInit);
            return;
        }
        if (params == 0u)
        {
            returnFromOverride(ctx, kNetBadParam);
            return;
        }
        int32_t type = 0;
        std::memcpy(&type, guestPtr(rdram, params), 4);
        int32_t backlog = 0;
        std::memcpy(&backlog, guestPtr(rdram, params + 4u), 4);
        if (netTrace())
            std::cerr << "[ATV2:NET] NetOpen type=" << type << " p1=" << backlog << " host='" << host << "' port=" << port
                      << " ra=0x" << std::hex << getRegU32(ctx, 31) << std::dec << std::endl;

        NetSocket ns;
        ns.type = type;
        int32_t status = kNetOk;
        if (type == 0)
        {
            in_addr ip{};
            if (host.empty() || port == 0u)
                status = kNetBadParam;
            else if (!resolveAndLog(host, ip))
                status = kNetError;
            else
            {
                ns.s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
                sockaddr_in to{};
                to.sin_family = AF_INET;
                to.sin_port = htons(port);
                to.sin_addr = ip;
                if (ns.s == kBadSocket || !setNonBlocking(ns.s) || !connectWithTimeout(ns.s, to))
                    status = kNetError;
                else
                {
                    int one = 1;
                    setsockopt(ns.s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char *>(&one), sizeof(one));
                }
                std::cerr << "[ATV2:NET] TCP connect " << host << " (" << ipToString(ip) << "):" << port << " -> "
                          << (status == kNetOk ? "ok" : "FAILED") << std::endl;
            }
        }
        else if (type == 1 || type == 2 || type == 3)
        {
            if (!host.empty() || (type != 3 && port == 0u) || (type == 1 && backlog < 1))
                status = kNetBadParam;
            else
            {
                ns.s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
                sockaddr_in me{};
                me.sin_family = AF_INET;
                me.sin_port = htons(type == 3 ? 0u : port);
                me.sin_addr.s_addr = htonl(INADDR_ANY);
                int one = 1;
                if (ns.s != kBadSocket)
                    setsockopt(ns.s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char *>(&one), sizeof(one));
#ifdef _WIN32
                if (ns.s != kBadSocket)
                {
                    // Don't surface ICMP port-unreachable as WSAECONNRESET on later recvfrom calls.
                    BOOL off = FALSE;
                    DWORD bytes = 0;
                    WSAIoctl(ns.s, SIO_UDP_CONNRESET, &off, sizeof(off), nullptr, 0, &bytes, nullptr, nullptr);
                }
#endif
                if (ns.s == kBadSocket || !setNonBlocking(ns.s) ||
                    bind(ns.s, reinterpret_cast<const sockaddr *>(&me), sizeof(me)) != 0)
                    status = kNetError;
                std::cerr << "[ATV2:NET] UDP bind type " << type << " port " << (type == 3 ? 0u : port)
                          << " -> " << (status == kNetOk ? "ok" : "FAILED") << std::endl;
            }
        }
        else
            status = kNetUnsupported;

        if (status != kNetOk)
        {
            if (ns.s != kBadSocket)
                closeHostSocket(ns.s);
            std::cerr << "[ATV2:NET] NetOpen type " << type << " failed, status " << status << std::endl;
            returnFromOverride(ctx, status);
            return;
        }
        const int32_t id = g_nextSocketId++;
        g_sockets[id] = ns;
        writeGuest32(rdram, outAddr, static_cast<uint32_t>(id));
        returnFromOverride(ctx, kNetOk);
    }

    // int NetClose(int sock) @ 0x0037E520
    void netClose(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        const int32_t id = static_cast<int32_t>(getRegU32(ctx, 4));
        std::lock_guard<std::mutex> lock(g_netMutex);
        if (!netInitialized(rdram))
        {
            returnFromOverride(ctx, kNetNotInit);
            return;
        }
        auto it = g_sockets.find(id);
        if (it == g_sockets.end())
        {
            returnFromOverride(ctx, id < 0 ? kNetBadParam : kNetError);
            return;
        }
        closeHostSocket(it->second.s);
        g_sockets.erase(it);
        if (netTrace())
        {
            std::cerr << "[ATV2:NET] close sock " << id << std::endl;
            atv2DumpWriteWatchRing();
        }
        returnFromOverride(ctx, kNetOk);
    }

    // int NetSend(int sock, const void *buf, u16 len, u16 *sent) @ 0x0037E570
    void netSend(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        const int32_t id = static_cast<int32_t>(getRegU32(ctx, 4));
        const uint32_t buf = getRegU32(ctx, 5);
        const uint16_t len = static_cast<uint16_t>(getRegU32(ctx, 6));
        const uint32_t sentAddr = getRegU32(ctx, 7);
        std::lock_guard<std::mutex> lock(g_netMutex);
        NetSocket *ns = findSocket(id);
        if (!netInitialized(rdram))
            return returnFromOverride(ctx, kNetNotInit);
        if (ns == nullptr || buf == 0u || sentAddr == 0u)
            return returnFromOverride(ctx, kNetBadParam);
        const int n = send(ns->s, reinterpret_cast<const char *>(guestPtr(rdram, buf)), len, 0);
        if (n < 0 && !wouldBlock(lastSockError()))
        {
            std::cerr << "[ATV2:NET] send sock " << id << " error " << lastSockError() << std::endl;
            return returnFromOverride(ctx, kNetError);
        }
        writeGuest16(rdram, sentAddr, static_cast<uint16_t>(n < 0 ? 0 : n));
        if (netTrace())
            std::cerr << "[ATV2:NET] send sock " << id << " " << len << " -> " << n
                      << hexDump(guestPtr(rdram, buf), n > 0 ? static_cast<size_t>(n) : 0u) << std::endl;
        returnFromOverride(ctx, kNetOk);
    }

    // int NetSendTo(int sock, const char *ip, u16 port, const void *buf, u16 len, u16 *sent) @ 0x0037E608
    void netSendTo(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        const int32_t id = static_cast<int32_t>(getRegU32(ctx, 4));
        const std::string ipStr = readGuestString(rdram, getRegU32(ctx, 5), 32);
        const uint16_t port = static_cast<uint16_t>(getRegU32(ctx, 6));
        const uint32_t buf = getRegU32(ctx, 7);
        const uint16_t len = static_cast<uint16_t>(getRegU32(ctx, 8));
        const uint32_t sentAddr = getRegU32(ctx, 9);
        std::lock_guard<std::mutex> lock(g_netMutex);
        NetSocket *ns = findSocket(id);
        if (!netInitialized(rdram))
            return returnFromOverride(ctx, kNetNotInit);
        if (ns == nullptr || ipStr.empty() || port == 0u || buf == 0u || sentAddr == 0u)
            return returnFromOverride(ctx, kNetBadParam);
        writeGuest16(rdram, sentAddr, 0u);
        sockaddr_in to{};
        to.sin_family = AF_INET;
        to.sin_port = htons(port);
        if (inet_pton(AF_INET, ipStr.c_str(), &to.sin_addr) != 1)
            return returnFromOverride(ctx, kNetError);
        const int n = sendto(ns->s, reinterpret_cast<const char *>(guestPtr(rdram, buf)), len, 0,
                             reinterpret_cast<const sockaddr *>(&to), sizeof(to));
        if (n < 0 && !wouldBlock(lastSockError()))
        {
            std::cerr << "[ATV2:NET] sendto sock " << id << " " << ipStr << ":" << port << " error " << lastSockError() << std::endl;
            return returnFromOverride(ctx, kNetError);
        }
        writeGuest16(rdram, sentAddr, static_cast<uint16_t>(n < 0 ? 0 : n));
        if (netTrace())
            std::cerr << "[ATV2:NET] sendto sock " << id << " " << ipStr << ":" << port << " " << len << " -> " << n << std::endl;
        returnFromOverride(ctx, kNetOk);
    }

    // int NetSendCheck(int sock, u8 *ok, const char *ip, u16 port) @ 0x0037E6F8
    // Retail posts a 2-byte probe record on the packet ring and reports whether the IOP accepted
    // it; host sockets can always queue a datagram.
    void netSendCheck(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        const uint32_t okAddr = getRegU32(ctx, 5);
        if (okAddr == 0u)
            return returnFromOverride(ctx, kNetBadParam);
        *guestPtr(rdram, okAddr) = 0u;
        if (!netInitialized(rdram))
            return returnFromOverride(ctx, kNetNotInit);
        *guestPtr(rdram, okAddr) = 1u;
        static int s_logged = 0;
        if (s_logged < 4)
        {
            ++s_logged;
            std::cerr << "[ATV2:NET] sendcheck sock " << static_cast<int32_t>(getRegU32(ctx, 4)) << " "
                      << readGuestString(rdram, getRegU32(ctx, 6), 32) << ":" << (getRegU32(ctx, 7) & 0xFFFFu) << std::endl;
        }
        returnFromOverride(ctx, kNetOk);
    }

    // int NetRecv(int sock, void *buf, u16 len, u16 *got) @ 0x0037E7B8; 0xD once the peer closed
    void netRecv(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        const int32_t id = static_cast<int32_t>(getRegU32(ctx, 4));
        const uint32_t buf = getRegU32(ctx, 5);
        const uint16_t len = static_cast<uint16_t>(getRegU32(ctx, 6));
        const uint32_t gotAddr = getRegU32(ctx, 7);
        std::lock_guard<std::mutex> lock(g_netMutex);
        NetSocket *ns = findSocket(id);
        if (!netInitialized(rdram))
            return returnFromOverride(ctx, kNetNotInit);
        if (ns == nullptr || buf == 0u || len > 0x1000u || gotAddr == 0u)
            return returnFromOverride(ctx, kNetBadParam);
        const int n = recv(ns->s, reinterpret_cast<char *>(guestPtr(rdram, buf)), len, 0);
        if (n == 0 && len != 0u)
        {
            std::cerr << "[ATV2:NET] recv sock " << id << ": peer closed" << std::endl;
            return returnFromOverride(ctx, kNetClosed);
        }
        if (n < 0)
        {
            const int e = lastSockError();
            if (!wouldBlock(e))
            {
                std::cerr << "[ATV2:NET] recv sock " << id << " error " << e << std::endl;
                return returnFromOverride(ctx, kNetClosed);
            }
        }
        writeGuest16(rdram, gotAddr, static_cast<uint16_t>(n < 0 ? 0 : n));
        if (netTrace() && n > 0)
            std::cerr << "[ATV2:NET] recv sock " << id << " -> " << n
                      << hexDump(guestPtr(rdram, buf), static_cast<size_t>(n)) << std::endl;
        returnFromOverride(ctx, kNetOk);
    }

    // int NetRecvFrom(int sock, char *ipOut, u16 *portOut, void *buf, u16 len, u16 *got) @ 0x0037E868
    void netRecvFrom(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        const int32_t id = static_cast<int32_t>(getRegU32(ctx, 4));
        const uint32_t ipOut = getRegU32(ctx, 5);
        const uint32_t portOut = getRegU32(ctx, 6);
        const uint32_t buf = getRegU32(ctx, 7);
        const uint16_t len = static_cast<uint16_t>(getRegU32(ctx, 8));
        const uint32_t gotAddr = getRegU32(ctx, 9);
        std::lock_guard<std::mutex> lock(g_netMutex);
        NetSocket *ns = findSocket(id);
        if (!netInitialized(rdram))
            return returnFromOverride(ctx, kNetNotInit);
        if (ns == nullptr || ipOut == 0u || portOut == 0u || buf == 0u || len > 0x1000u || gotAddr == 0u)
            return returnFromOverride(ctx, kNetBadParam);
        writeGuest16(rdram, gotAddr, 0u);
        sockaddr_in from{};
        socklen_t fromLen = sizeof(from);
        const int n = recvfrom(ns->s, reinterpret_cast<char *>(guestPtr(rdram, buf)), len, 0,
                               reinterpret_cast<sockaddr *>(&from), &fromLen);
        if (n < 0)
        {
            const int e = lastSockError();
#ifdef _WIN32
            if (!wouldBlock(e) && e != WSAECONNRESET && e != WSAEMSGSIZE)
#else
            if (!wouldBlock(e))
#endif
            {
                std::cerr << "[ATV2:NET] recvfrom sock " << id << " error " << e << std::endl;
                return returnFromOverride(ctx, kNetError);
            }
            return returnFromOverride(ctx, kNetOk);
        }
        writeGuest16(rdram, portOut, ntohs(from.sin_port));
        writeGuestString(rdram, ipOut, ipToString(from.sin_addr), 16u);
        writeGuest16(rdram, gotAddr, static_cast<uint16_t>(n));
        if (netTrace())
            std::cerr << "[ATV2:NET] recvfrom sock " << id << " " << ipToString(from.sin_addr) << ":"
                      << ntohs(from.sin_port) << " -> " << n << std::endl;
        returnFromOverride(ctx, kNetOk);
    }

    // int NetPollAccept(int sock, u8 *flag, char *ipOut, u16 *portOut) @ 0x0037E970
    // Retail reads a status-2 record (incoming connection) off a listen socket's packet ring. The
    // NetLib API has no way to hand back the accepted socket, so report "nothing pending".
    void netPollAccept(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        const uint32_t flagAddr = getRegU32(ctx, 5);
        if (flagAddr == 0u)
            return returnFromOverride(ctx, kNetBadParam);
        *guestPtr(rdram, flagAddr) = 0u;
        static bool s_logged = false;
        if (!s_logged)
        {
            s_logged = true;
            std::cerr << "[ATV2:NET] pollaccept sock " << static_cast<int32_t>(getRegU32(ctx, 4)) << " (unsupported, none)" << std::endl;
        }
        returnFromOverride(ctx, netInitialized(rdram) ? kNetOk : kNetNotInit);
    }

    // int NetAvailable(int sock, u16 *count) @ 0x0037EA40; 0xD once the peer closed
    void netAvailable(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        const int32_t id = static_cast<int32_t>(getRegU32(ctx, 4));
        const uint32_t countAddr = getRegU32(ctx, 5);
        std::lock_guard<std::mutex> lock(g_netMutex);
        NetSocket *ns = findSocket(id);
        if (!netInitialized(rdram))
            return returnFromOverride(ctx, kNetNotInit);
        if (ns == nullptr || countAddr == 0u)
            return returnFromOverride(ctx, kNetBadParam);
        int n = bytesPending(ns->s);
        if (n == 0 && ns->type == 0)
        {
            fd_set rd;
            FD_ZERO(&rd);
            FD_SET(ns->s, &rd);
            timeval tv{0, 0};
            char c;
            if (select(static_cast<int>(ns->s) + 1, &rd, nullptr, nullptr, &tv) > 0 && recv(ns->s, &c, 1, MSG_PEEK) == 0)
                return returnFromOverride(ctx, kNetClosed);
        }
        writeGuest16(rdram, countAddr, static_cast<uint16_t>(n < 0 ? 0 : std::min(n, 0xFFFF)));
        returnFromOverride(ctx, kNetOk);
    }

    // int NetGetLocalIp(char *out) @ 0x0037EAD8 / NetGetDnsIp(char *out) @ 0x0037EB20
    void netGetLocalIp(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        if (!netInitialized(rdram))
            return returnFromOverride(ctx, kNetNotInit);
        if (getRegU32(ctx, 4) == 0u)
            return returnFromOverride(ctx, kNetBadParam);
        writeGuestString(rdram, getRegU32(ctx, 4), g_localIp, 16u);
        returnFromOverride(ctx, kNetOk);
    }

    void netGetDnsIp(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        if (!netInitialized(rdram))
            return returnFromOverride(ctx, kNetNotInit);
        if (getRegU32(ctx, 4) == 0u)
            return returnFromOverride(ctx, kNetBadParam);
        writeGuestString(rdram, getRegU32(ctx, 4), dnsServer(), 16u);
        returnFromOverride(ctx, kNetOk);
    }

    // int NetResolve(const char *host, char *ipOut) @ 0x0037EB68
    void netResolve(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        const uint32_t hostAddr = getRegU32(ctx, 4);
        const uint32_t outAddr = getRegU32(ctx, 5);
        if (!netInitialized(rdram))
            return returnFromOverride(ctx, kNetNotInit);
        if (hostAddr == 0u || outAddr == 0u)
            return returnFromOverride(ctx, kNetBadParam);
        in_addr ip{};
        if (!resolveAndLog(readGuestString(rdram, hostAddr), ip))
            return returnFromOverride(ctx, kNetError);
        writeGuestString(rdram, outAddr, ipToString(ip), 16u);
        returnFromOverride(ctx, kNetOk);
    }

    // int NetGetPeerIp(int sock, char *ipOut) @ 0x0037EC18
    void netGetPeerIp(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        const int32_t id = static_cast<int32_t>(getRegU32(ctx, 4));
        const uint32_t outAddr = getRegU32(ctx, 5);
        std::lock_guard<std::mutex> lock(g_netMutex);
        NetSocket *ns = findSocket(id);
        if (!netInitialized(rdram))
            return returnFromOverride(ctx, kNetNotInit);
        if (ns == nullptr || outAddr == 0u)
            return returnFromOverride(ctx, kNetBadParam);
        sockaddr_in peer{};
        socklen_t peerLen = sizeof(peer);
        if (getpeername(ns->s, reinterpret_cast<sockaddr *>(&peer), &peerLen) != 0)
            return returnFromOverride(ctx, kNetError);
        writeGuestString(rdram, outAddr, ipToString(peer.sin_addr), 16u);
        returnFromOverride(ctx, kNetOk);
    }

    // int NetTick(void) @ 0x0037ECC8 (retail: msifrpc call 0x67 to service the IOP side)
    void netTick(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        returnFromOverride(ctx, netInitialized(rdram) ? kNetOk : kNetNotInit);
    }

    // ATV2_NET_TRACE: log the first calls of the quiet NetLib queries with arguments and status.
    template <PS2Runtime::RecompiledFunction F, int Id>
    void netQueryTraced(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        static const char *kNames[] = {"NetAvailable", "NetGetLocalIp", "NetGetDnsIp", "NetGetPeerIp"};
        static int s_calls = 0;
        const uint32_t a0 = getRegU32(ctx, 4), a1 = getRegU32(ctx, 5), ra = getRegU32(ctx, 31);
        F(rdram, ctx, runtime);
        if (netTrace() && s_calls < 6)
        {
            ++s_calls;
            std::cerr << "[ATV2:NET] " << kNames[Id] << "(0x" << std::hex << a0 << ", 0x" << a1 << ") -> "
                      << std::dec << static_cast<int32_t>(getRegU32(ctx, 2)) << " ra=0x" << std::hex << ra << std::dec;
            if (Id != 0 && a0 != 0u)
                std::cerr << " out='" << readGuestString(rdram, Id == 3 ? a1 : a0, 24) << "'";
            std::cerr << std::endl;
        }
    }

    // int PSXNetwork::LookupHost(const char *host, char *ipOut) @ 0x00304660 (vtable 0x45F910 slot)
    // Retail binds its own libnet client, runs Name2Address + Address2String, and always returns 1.
    void psxNetworkLookupHost(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        const std::string host = readGuestString(rdram, getRegU32(ctx, 5));
        in_addr ip{};
        const bool ok = resolveAndLog(host, ip);
        writeGuestString(rdram, getRegU32(ctx, 6), ok ? ipToString(ip) : std::string("0.0.0.0"), 16u);
        returnFromOverride(ctx, 1);
    }

    // u32 RtGetCpuCount(void) @ 0x0037F908: SCE-RT's clock is `mfc0 Count` (294.912 MHz), but the
    // runtime never advances ctx->cop0_count, so Medius timers froze and login never progressed.
    // Return the host monotonic clock in EE cycles instead.
    void rtCpuCount(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        static const auto s_start = std::chrono::steady_clock::now();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - s_start).count();
        const uint64_t cycles = static_cast<uint64_t>(ns) * 294912ull / 1000000ull;
        returnFromOverride(ctx, static_cast<int32_t>(static_cast<uint32_t>(cycles)));
    }

    // const char *GetRomVersion(void) @ 0x00363608 (libscf): lazily reads 14 bytes of rom0:ROMVER into
    // 0x004121A0. SCE-RT's clock code reaches it (sceScfGetLocalTimefromRTC -> IsT10K) right after the
    // network comes up, and the runtime's rom0: file path crashes the host. Supply a retail US
    // console's version string instead (byte 4 'A' = not a T10K devkit).
    void romVersion(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        constexpr uint32_t kRomVerAddr = 0x004121A0u;
        if (*guestPtr(rdram, kRomVerAddr) == 0u)
            std::memcpy(guestPtr(rdram, kRomVerAddr), "0160AC20020207", 14);
        returnFromOverride(ctx, static_cast<int32_t>(kRomVerAddr));
    }

    // Guards: everything that reaches libnet's msifrpc path should have been routed above. Log and
    // fail instead of spinning forever on an IOP server that doesn't exist.
    void libnetBindGuard(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        std::cerr << "[ATV2:NET] WARNING: un-HLE'd libnet bind (client 0x" << std::hex << getRegU32(ctx, 4)
                  << ") from ra=0x" << getRegU32(ctx, 31) << std::dec << std::endl;
        (void)rdram;
        returnFromOverride(ctx, -1);
    }

    void msifCallRpcGuard(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        static int s_logged = 0;
        if (s_logged < 16)
        {
            ++s_logged;
            std::cerr << "[ATV2:NET] WARNING: un-HLE'd sceSifMCallRpc fno=0x" << std::hex << getRegU32(ctx, 5)
                      << " ra=0x" << getRegU32(ctx, 31) << std::dec << std::endl;
        }
        returnFromOverride(ctx, -1);
    }

    // ATV2_NET_TRACE: log UI alerts (QuadzillaUIPage alert setup @ 0x0025E120, a1 = alert id) so
    // the front end's network gates can be traced to their callers.
    PS2Runtime::RecompiledFunction g_origUiAlert = nullptr;
    void uiAlertTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        std::cerr << "[ATV2:NET] UI alert id=0x" << std::hex << getRegU32(ctx, 5) << " page=0x" << getRegU32(ctx, 4)
                  << " ra=0x" << getRegU32(ctx, 31) << std::dec << std::endl;
        g_origUiAlert(rdram, ctx, runtime);
    }

    // ATV2_NET_TRACE: QuadzillaUIPageProfileStore_Menu input handler @ 0x002A1B40 (also the
    // "Select a Network Configuration" page): a1 = repeat flag, a2 = input code (0x2A accept).
    PS2Runtime::RecompiledFunction g_origProfileStoreInput = nullptr;
    void profileStoreInputTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t page = getRegU32(ctx, 4);
        uint32_t alert = 0, busy = 0, block = 0;
        std::memcpy(&alert, guestPtr(rdram, page + 0x428u), 4);
        std::memcpy(&busy, guestPtr(rdram, page + 0x578u), 4);
        std::memcpy(&block, guestPtr(rdram, page + 0x550u), 4);
        static int s_calls = 0;
        if (getRegU32(ctx, 6) != 0u || (++s_calls % 120) == 1)
            std::cerr << "[ATV2:NET] ProfileStore input page=0x" << std::hex << page << " a1=0x" << getRegU32(ctx, 5)
                      << " code=0x" << getRegU32(ctx, 6) << " alert=0x" << alert << " busy=" << busy << " block=" << block
                      << std::dec << std::endl;
        g_origProfileStoreInput(rdram, ctx, runtime);
    }

    // ATV2_NET_TRACE: Multiplayer Mode page input handler @ 0x00265FD0 (same slot 0xC8 signature).
    PS2Runtime::RecompiledFunction g_origMultiplayerInput = nullptr;
    void multiplayerInputTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (getRegU32(ctx, 6) != 0u)
            std::cerr << "[ATV2:NET] Multiplayer input page=0x" << std::hex << getRegU32(ctx, 4) << " a1=0x"
                      << getRegU32(ctx, 5) << " code=0x" << getRegU32(ctx, 6) << std::dec << std::endl;
        g_origMultiplayerInput(rdram, ctx, runtime);
    }

    // ATV2_NET_TRACE: guest printf @ 0x0036DD28 (the network libraries log through it).
    PS2Runtime::RecompiledFunction g_origPrintf = nullptr;
    void printfTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        std::string fmt = readGuestString(rdram, getRegU32(ctx, 4), 120);
        while (!fmt.empty() && (fmt.back() == '\n' || fmt.back() == '\r'))
            fmt.pop_back();
        std::cerr << "[ATV2:NET] printf ra=0x" << std::hex << getRegU32(ctx, 31) << " a1=0x" << getRegU32(ctx, 5)
                  << std::dec << " \"" << fmt << "\"";
        if (fmt.find("%s") != std::string::npos)
            std::cerr << " s1=\"" << readGuestString(rdram, getRegU32(ctx, 5), 80) << "\"";
        std::cerr << std::endl;
        g_origPrintf(rdram, ctx, runtime);
    }

    // ATV2_NET_TRACE: SCE-RT cipher key schedule @ 0x00382C80 (ctx, key[64], hash u32). Logs each
    // distinct session key so captured RT traffic can be decrypted offline.
    PS2Runtime::RecompiledFunction g_origRtSetKey = nullptr;
    void rtSetKeyTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        static std::string s_lastKey;
        const std::string key = hexDump(guestPtr(rdram, getRegU32(ctx, 5)), 64u);
        if (key != s_lastKey)
        {
            s_lastKey = key;
            std::cerr << "[ATV2:NET] rt key ctx=0x" << std::hex << getRegU32(ctx, 4) << " ra=0x" << getRegU32(ctx, 31)
                      << std::dec << key << std::endl;
        }
        g_origRtSetKey(rdram, ctx, runtime);
    }

    // ATV2_NET_TRACE: MGCL connect-result callback @ 0x003B1F18 (a1 = 1 on success, a3 = net error).
    PS2Runtime::RecompiledFunction g_origMgclConnectCb = nullptr;
    void mgclConnectCbTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        std::cerr << "[ATV2:NET] MGCL connect result a0=0x" << std::hex << getRegU32(ctx, 4) << std::dec
                  << " status=" << static_cast<int32_t>(getRegU32(ctx, 5)) << " a2=" << static_cast<int32_t>(getRegU32(ctx, 6))
                  << " netError=" << static_cast<int32_t>(getRegU32(ctx, 7)) << " ra=0x" << std::hex << getRegU32(ctx, 31)
                  << std::dec << std::endl;
        g_origMgclConnectCb(rdram, ctx, runtime);
    }

    // ATV2_NET_TRACE: MGCL request after server session begin @ 0x003B01C0 (used by the lobby
    // manager's step 3); a non-zero return fails the online login. MGCL connection state
    // getter @ 0x003AFD88.
    PS2Runtime::RecompiledFunction g_origMgclRequest = nullptr;
    PS2Runtime::RecompiledFunction g_origMgclState = nullptr;
    void mgclStateTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        g_origMgclState(rdram, ctx, runtime);
        static int s_logged = 0;
        if (s_logged < 12)
        {
            ++s_logged;
            std::cerr << "[ATV2:NET] MGCL state -> " << static_cast<int32_t>(getRegU32(ctx, 2)) << std::endl;
        }
    }
    void mgclRequestTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t ra = getRegU32(ctx, 31);
        g_origMgclRequest(rdram, ctx, runtime);
        std::cerr << "[ATV2:NET] MGCL request 0x3B01C0 -> " << static_cast<int32_t>(getRegU32(ctx, 2)) << " ra=0x"
                  << std::hex << ra << std::dec << std::endl;
    }

    // ATV2_NET_TRACE: lobby-manager step functions (0x317990 connect, 0x317B10 server session begin,
    // 0x317D60 next MGCL request) and the session-begin callback 0x30A5F0. Logs a line whenever a
    // step's return value or the manager's sub-state bytes (+0x122..+0x127) change.
    PS2Runtime::RecompiledFunction g_origLobbyStep[3] = {};
    template <int I>
    void lobbyStepTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        static const char *kNames[3] = {"connect 0x317990", "sessionBegin 0x317B10", "request 0x317D60"};
        static std::string s_last;
        const uint32_t obj = getRegU32(ctx, 4);
        g_origLobbyStep[I](rdram, ctx, runtime);
        std::string now = std::to_string(static_cast<int32_t>(getRegU32(ctx, 2))) + " sub=" +
                          hexDump(guestPtr(rdram, obj + 0x122u), 6u);
        if (now != s_last)
        {
            s_last = now;
            std::cerr << "[ATV2:NET] lobby step " << kNames[I] << " -> " << now << std::endl;
        }
    }
    // MGCL session-begin reply handler @ 0x003B00D8 (a3 = parsed reply; +0x15 confirmation).
    PS2Runtime::RecompiledFunction g_origSessionBeginHandler = nullptr;
    void sessionBeginHandlerTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t resp = getRegU32(ctx, 7);
        uint32_t cb = 0;
        std::memcpy(&cb, guestPtr(rdram, 0x0041D4C8u), 4);
        std::cerr << "[ATV2:NET] MGCL session-begin handler resp=0x" << std::hex << resp << " cb=0x" << cb << std::dec
                  << (resp ? hexDump(guestPtr(rdram, resp), 0x30u) : std::string()) << std::endl;
        g_origSessionBeginHandler(rdram, ctx, runtime);
    }
    PS2Runtime::RecompiledFunction g_origSessionBeginCb = nullptr;
    void sessionBeginCbTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t resp = getRegU32(ctx, 7);
        std::cerr << "[ATV2:NET] MGCL session-begin callback resp=0x" << std::hex << resp << std::dec
                  << (resp ? hexDump(guestPtr(rdram, resp), 0x30u) : std::string()) << std::endl;
        g_origSessionBeginCb(rdram, ctx, runtime);
    }
    // Join game (lobby page, alert 0x20 "Game is full!" on any failure): 0x003168F0(loginMgr, gameId)
    // = Medius JoinGame (callback 0x0030A160: a3 = reply, +0x18 status, +0x28 host address) then the
    // DME connect 0x0039A0A0 and a 30 s wait for the host.
    PS2Runtime::RecompiledFunction g_origJoinGame = nullptr;
    void joinGameTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        std::cerr << "[ATV2:NET] join game id=" << static_cast<int32_t>(getRegU32(ctx, 5)) << std::endl;
        g_origJoinGame(rdram, ctx, runtime);
        std::cerr << "[ATV2:NET] join game -> " << static_cast<int32_t>(getRegU32(ctx, 2)) << std::endl;
    }
    PS2Runtime::RecompiledFunction g_origJoinGameCb = nullptr;
    void joinGameCbTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t resp = getRegU32(ctx, 7);
        int32_t status = 0;
        if (resp)
            std::memcpy(&status, guestPtr(rdram, resp + 0x18u), 4);
        std::cerr << "[ATV2:NET] Medius JoinGame reply status=" << status << " resp=0x" << std::hex << resp << std::dec
                  << (resp ? hexDump(guestPtr(rdram, resp), 0x60u) : std::string()) << std::endl;
        g_origJoinGameCb(rdram, ctx, runtime);
    }
    PS2Runtime::RecompiledFunction g_origDmeConnect = nullptr;
    void dmeConnectTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t req = getRegU32(ctx, 4);
        std::cerr << "[ATV2:NET] DME connect req=0x" << std::hex << req << std::dec
                  << (req ? hexDump(guestPtr(rdram, req), 0xC0u) : std::string()) << std::endl;
        g_origDmeConnect(rdram, ctx, runtime);
        std::cerr << "[ATV2:NET] DME connect -> " << static_cast<int32_t>(getRegU32(ctx, 2)) << std::endl;
    }

    // ATV2_NET_TRACE: countdown helper @ 0x003032B0 (timer struct: float remaining, u64 last ms at +8).
    PS2Runtime::RecompiledFunction g_origCountdown = nullptr;
    void countdownTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t t = getRegU32(ctx, 4);
        float before = 0.f;
        uint32_t lastBefore = 0;
        std::memcpy(&before, guestPtr(rdram, t), 4);
        std::memcpy(&lastBefore, guestPtr(rdram, t + 8u), 4);
        g_origCountdown(rdram, ctx, runtime);
        float after = 0.f;
        uint32_t lastAfter = 0;
        std::memcpy(&after, guestPtr(rdram, t), 4);
        std::memcpy(&lastAfter, guestPtr(rdram, t + 8u), 4);
        static int s_logged = 0;
        if (s_logged < 2000 && (after <= 1.0f || (s_logged % 20) == 0 || before - after > 1.0f))
        {
            ++s_logged;
            std::cerr << "[ATV2:NET] countdown t=0x" << std::hex << t << std::dec << " " << before << " -> " << after
                      << " lastMs " << lastBefore << " -> " << lastAfter << " ra=0x" << std::hex << getRegU32(ctx, 31)
                      << std::dec << std::endl;
        }
    }

    // ATV2_NET_TRACE: guest newlib sprintf @ 0x0036F400 (a0 = dst, a1 = format, a2.. = args).
    PS2Runtime::RecompiledFunction g_origSprintf = nullptr;
    void sprintfTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        static int s_logged = 0;
        if (s_logged < 400)
        {
            ++s_logged;
            const std::string fmt = readGuestString(rdram, getRegU32(ctx, 5), 100);
            std::cerr << "[ATV2:NET] sprintf ra=0x" << std::hex << getRegU32(ctx, 31) << " dst=0x" << getRegU32(ctx, 4)
                      << " a2=0x" << getRegU32(ctx, 6) << " a3=0x" << getRegU32(ctx, 7) << std::dec << " fmt=\"" << fmt
                      << "\"" << std::endl;
        }
        g_origSprintf(rdram, ctx, runtime);
    }

    // ATV2_NET_TRACE: flag oversized guest memcpy/memmove/memset (a2 = length) with their caller.
    PS2Runtime::RecompiledFunction g_origMem[3] = {};
    template <int I>
    void bigMemTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        static const char *kNames[3] = {"memcpy", "memmove", "memset"};
        if (getRegU32(ctx, 6) >= 0x100000u)
            std::cerr << "[ATV2:NET] big " << kNames[I] << " dst=0x" << std::hex << getRegU32(ctx, 4) << " src=0x"
                      << getRegU32(ctx, 5) << " len=0x" << getRegU32(ctx, 6) << " ra=0x" << getRegU32(ctx, 31)
                      << " sp=0x" << getRegU32(ctx, 29) << std::dec << std::endl;
        g_origMem[I](rdram, ctx, runtime);
    }

    // ATV2_NET_TRACE: base page input handler @ 0x0025C510 (page, a1, a2 = code).
    PS2Runtime::RecompiledFunction g_origBasePageInput = nullptr;
    void basePageInputTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (getRegU32(ctx, 6) != 0u)
            std::cerr << "[ATV2:NET] base page input page=0x" << std::hex << getRegU32(ctx, 4) << " a1=0x"
                      << getRegU32(ctx, 5) << " code=0x" << getRegU32(ctx, 6) << " ra=0x" << getRegU32(ctx, 31)
                      << std::dec << std::endl;
        g_origBasePageInput(rdram, ctx, runtime);
    }

    // ATV2_NET_TRACE: network-config list fill @ 0x002A1690 (returns the first valid row or -1).
    PS2Runtime::RecompiledFunction g_origNetcnfListFill = nullptr;
    void netcnfListFillTrace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t page = getRegU32(ctx, 4);
        const uint32_t ra = getRegU32(ctx, 31);
        std::cerr << "[ATV2:NET] netcnf list fill enter page=0x" << std::hex << page << " ra=0x" << ra << std::dec << std::endl;
        g_origNetcnfListFill(rdram, ctx, runtime);
        uint32_t alert = 0;
        std::memcpy(&alert, guestPtr(rdram, page + 0x428u), 4);
        std::cerr << "[ATV2:NET] netcnf list fill exit v0=" << static_cast<int32_t>(getRegU32(ctx, 2)) << " pc=0x" << std::hex
                  << ctx->pc << " alert=0x" << alert << std::dec << std::endl;
    }

    struct NetHleEntry
    {
        uint32_t addr;
        PS2Runtime::RecompiledFunction fn;
    };

    const NetHleEntry kNetHleEntries[] = {
        // ezNetCnf client (SIF RPC 0x75499128)
        {0x003B1FC8u, &ezNetCnfBind},
        {0x003B2090u, &ezNetCnfUnbind},
        {0x003B2118u, &ezNetCnfGetCount},
        {0x003B21D8u, &ezNetCnfGetCombinationList},
        {0x003B2398u, &ezNetCnfSetDefault},
        {0x003B2460u, &ezNetCnfGetNetcnfifData},
        // ezNetCtl client (SIF RPC 0x75488909)
        {0x003B2538u, &ezNetCtlBind},
        {0x003B2600u, &ezNetCtlUnbind},
        {0x003B2688u, &ezNetCtlSetProperties},
        {0x003B27A8u, &ezNetCtlSetCombination},
        {0x003B2870u, &ezNetCtlGetStatus},
        {0x003B2988u, &ezNetCtlCommand},
        {0x003B2A00u, &ezNetCtlCommand},
        // NetLib platform layer: thunk table entries and their implementations
        {0x0037DF98u, &netInit},
        {0x0037E120u, &netInit},
        {0x0037DFB0u, &netShutdown},
        {0x0037E1E8u, &netShutdown},
        {0x0037DFC8u, &netOpen},
        {0x0037E268u, &netOpen},
        {0x0037DFE0u, &netClose},
        {0x0037E520u, &netClose},
        {0x0037DFF8u, &netSend},
        {0x0037E570u, &netSend},
        {0x0037E010u, &netSendTo},
        {0x0037E608u, &netSendTo},
        {0x0037E030u, &netSendCheck},
        {0x0037E6F8u, &netSendCheck},
        {0x0037E048u, &netRecv},
        {0x0037E7B8u, &netRecv},
        {0x0037E060u, &netRecvFrom},
        {0x0037E868u, &netRecvFrom},
        {0x0037E078u, &netPollAccept},
        {0x0037E970u, &netPollAccept},
        {0x0037E090u, &netQueryTraced<&netAvailable, 0>},
        {0x0037EA40u, &netQueryTraced<&netAvailable, 0>},
        {0x0037E0A8u, &netQueryTraced<&netGetLocalIp, 1>},
        {0x0037EAD8u, &netQueryTraced<&netGetLocalIp, 1>},
        {0x0037E0C0u, &netQueryTraced<&netGetDnsIp, 2>},
        {0x0037EB20u, &netQueryTraced<&netGetDnsIp, 2>},
        {0x0037E0D8u, &netResolve},
        {0x0037EB68u, &netResolve},
        {0x0037E0F0u, &netQueryTraced<&netGetPeerIp, 3>},
        {0x0037EC18u, &netQueryTraced<&netGetPeerIp, 3>},
        {0x0037E108u, &netTick},
        {0x0037ECC8u, &netTick},
        // PSXNetwork host lookup
        {0x00304660u, &psxNetworkLookupHost},
        // SCE-RT clock source
        {0x0037F908u, &rtCpuCount},
        // libscf ROM version (rom0:ROMVER)
        {0x00363608u, &romVersion},
        // guards
        {0x0037B030u, &libnetBindGuard},
        {0x003642C8u, &msifCallRpcGuard},
    };
}

void registerAtv2NetworkHle(PS2Runtime &runtime)
{
    const char *v = std::getenv("ATV2_NET_HLE");
    if (v != nullptr && v[0] == '0')
        return;
    for (const NetHleEntry &e : kNetHleEntries)
        runtime.replaceFunction(e.addr, e.fn);
#ifdef _WIN32
    if (netTrace())
        AddVectoredExceptionHandler(0, &netCrashHandler);
#endif
    if (netTrace() && runtime.hasFunction(0x0025E120u))
    {
        g_origUiAlert = runtime.lookupFunction(0x0025E120u);
        runtime.replaceFunction(0x0025E120u, &uiAlertTrace);
    }
    if (netTrace() && runtime.hasFunction(0x002A1B40u))
    {
        g_origProfileStoreInput = runtime.lookupFunction(0x002A1B40u);
        runtime.replaceFunction(0x002A1B40u, &profileStoreInputTrace);
    }
    else if (netTrace())
        std::cerr << "[ATV2:NET] no function entry at 0x2A1B40 (input trace off)" << std::endl;
    if (netTrace())
    {
        const uint32_t memFns[3] = {0x0036C968u, 0x0036CA18u, 0x0036CB20u};
        const PS2Runtime::RecompiledFunction wrappers[3] = {&bigMemTrace<0>, &bigMemTrace<1>, &bigMemTrace<2>};
        for (int i = 0; i < 3; ++i)
            if (runtime.hasFunction(memFns[i]))
            {
                g_origMem[i] = runtime.lookupFunction(memFns[i]);
                runtime.replaceFunction(memFns[i], wrappers[i]);
            }
    }
    if (netTrace())
    {
        const uint32_t steps[3] = {0x00317990u, 0x00317B10u, 0x00317D60u};
        const PS2Runtime::RecompiledFunction wraps[3] = {&lobbyStepTrace<0>, &lobbyStepTrace<1>, &lobbyStepTrace<2>};
        for (int i = 0; i < 3; ++i)
            if (runtime.hasFunction(steps[i]))
            {
                g_origLobbyStep[i] = runtime.lookupFunction(steps[i]);
                runtime.replaceFunction(steps[i], wraps[i]);
            }
        if (runtime.hasFunction(0x0030A5F0u))
        {
            g_origSessionBeginCb = runtime.lookupFunction(0x0030A5F0u);
            runtime.replaceFunction(0x0030A5F0u, &sessionBeginCbTrace);
        }
        else
            std::cerr << "[ATV2:NET] no function entry at 0x30A5F0 (callback trace off)" << std::endl;
        if (runtime.hasFunction(0x003B00D8u))
        {
            g_origSessionBeginHandler = runtime.lookupFunction(0x003B00D8u);
            runtime.replaceFunction(0x003B00D8u, &sessionBeginHandlerTrace);
        }
        else
            std::cerr << "[ATV2:NET] no function entry at 0x3B00D8 (handler trace off)" << std::endl;
        const struct
        {
            uint32_t addr;
            PS2Runtime::RecompiledFunction *orig;
            PS2Runtime::RecompiledFunction wrap;
        } joinTraces[] = {{0x003168F0u, &g_origJoinGame, &joinGameTrace},
                          {0x0030A160u, &g_origJoinGameCb, &joinGameCbTrace},
                          {0x0039A0A0u, &g_origDmeConnect, &dmeConnectTrace}};
        for (const auto &t : joinTraces)
            if (runtime.hasFunction(t.addr))
            {
                *t.orig = runtime.lookupFunction(t.addr);
                runtime.replaceFunction(t.addr, t.wrap);
            }
            else
                std::cerr << "[ATV2:NET] no function entry at 0x" << std::hex << t.addr << std::dec << " (join trace off)"
                          << std::endl;
    }
    if (netTrace() && runtime.hasFunction(0x003032B0u))
    {
        g_origCountdown = runtime.lookupFunction(0x003032B0u);
        runtime.replaceFunction(0x003032B0u, &countdownTrace);
    }
    if (netTrace() && runtime.hasFunction(0x003B01C0u) && runtime.hasFunction(0x003AFD88u))
    {
        g_origMgclRequest = runtime.lookupFunction(0x003B01C0u);
        runtime.replaceFunction(0x003B01C0u, &mgclRequestTrace);
        g_origMgclState = runtime.lookupFunction(0x003AFD88u);
        runtime.replaceFunction(0x003AFD88u, &mgclStateTrace);
    }
    if (netTrace() && runtime.hasFunction(0x003B1F18u))
    {
        g_origMgclConnectCb = runtime.lookupFunction(0x003B1F18u);
        runtime.replaceFunction(0x003B1F18u, &mgclConnectCbTrace);
    }
    if (netTrace() && runtime.hasFunction(0x00382C80u))
    {
        g_origRtSetKey = runtime.lookupFunction(0x00382C80u);
        runtime.replaceFunction(0x00382C80u, &rtSetKeyTrace);
    }
    if (netTrace() && runtime.hasFunction(0x0036F400u))
    {
        g_origSprintf = runtime.lookupFunction(0x0036F400u);
        runtime.replaceFunction(0x0036F400u, &sprintfTrace);
    }
    if (netTrace() && runtime.hasFunction(0x0036DD28u))
    {
        g_origPrintf = runtime.lookupFunction(0x0036DD28u);
        runtime.replaceFunction(0x0036DD28u, &printfTrace);
    }
    if (netTrace() && runtime.hasFunction(0x0025C510u))
    {
        g_origBasePageInput = runtime.lookupFunction(0x0025C510u);
        runtime.replaceFunction(0x0025C510u, &basePageInputTrace);
    }
    if (netTrace() && runtime.hasFunction(0x00265FD0u))
    {
        g_origMultiplayerInput = runtime.lookupFunction(0x00265FD0u);
        runtime.replaceFunction(0x00265FD0u, &multiplayerInputTrace);
    }
    if (netTrace() && runtime.hasFunction(0x002A1690u))
    {
        g_origNetcnfListFill = runtime.lookupFunction(0x002A1690u);
        runtime.replaceFunction(0x002A1690u, &netcnfListFillTrace);
    }
    std::cerr << "[ATV2:NET] network HLE registered (dns " << dnsServer() << ")" << std::endl;
}
