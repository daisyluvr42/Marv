import Darwin
import Testing
@testable import MarvDiscovery

@Suite
struct WideAreaGatewayDiscoveryTests {
    @Test func discoversBeaconFromTailnetDnsSdFallback() {
        setenv("OPENCLAW_WIDE_AREA_DOMAIN", "marv.internal", 1)
        let statusJson = """
        {
          "Self": { "TailscaleIPs": ["100.69.232.64"] },
          "Peer": {
            "peer-1": { "TailscaleIPs": ["100.123.224.76"] }
          }
        }
        """

        let context = WideAreaGatewayDiscovery.DiscoveryContext(
            tailscaleStatus: { statusJson },
            dig: { args, _ in
                let recordType = args.last ?? ""
                let nameserver = args.first(where: { $0.hasPrefix("@") }) ?? ""
                if recordType == "PTR" {
                    if nameserver == "@100.123.224.76" {
                        return "monadlabstudio-gateway._marv-gw._tcp.marv.internal.\n"
                    }
                    return ""
                }
                if recordType == "SRV" {
                    return "0 0 4242 monadlabstudio.marv.internal."
                }
                if recordType == "TXT" {
                    return "\"displayName=Monad Lab\\226\\128\\153s Mac Studio (Marv)\" \"gatewayPort=4242\" \"tailnetDns=monad-lab-studio-1.sheep-coho.ts.net\" \"cliPath=/Users/monadlab/marv/src/entry.ts\""
                }
                return ""
            })

        let beacons = WideAreaGatewayDiscovery.discover(
            timeoutSeconds: 2.0,
            context: context)

        #expect(beacons.count == 1)
        let beacon = beacons[0]
        let expectedDisplay = "Monad Lab\u{2019}s Mac Studio (Marv)"
        #expect(beacon.displayName == expectedDisplay)
        #expect(beacon.port == 4242)
        #expect(beacon.gatewayPort == 4242)
        #expect(beacon.tailnetDns == "monad-lab-studio-1.sheep-coho.ts.net")
        #expect(beacon.cliPath == "/Users/monadlab/marv/src/entry.ts")
    }
}
