import OpenClawChatUI
import OpenClawProtocol
import Testing
@testable import Marv

@Suite struct MacGatewayChatTransportMappingTests {
    @Test func snapshotMapsToHealth() {
        let snapshot = Snapshot(
            presence: [],
            health: OpenClawProtocol.AnyCodable(["ok": OpenClawProtocol.AnyCodable(false)]),
            stateversion: StateVersion(presence: 1, health: 1),
            uptimems: 123,
            configpath: nil,
            statedir: nil,
            sessiondefaults: nil,
            authmode: nil,
            updateavailable: nil)

        let hello = HelloOk(
            type: "hello",
            _protocol: 2,
            server: [:],
            features: [:],
            snapshot: snapshot,
            canvashosturl: nil,
            auth: nil,
            policy: [:])

        let mapped = MacGatewayChatTransport.mapPushToTransportEvent(.snapshot(hello))
        switch mapped {
        case let .health(ok):
            #expect(ok == false)
        default:
            Issue.record("expected .health from snapshot, got \(String(describing: mapped))")
        }
    }

    @Test func healthEventMapsToHealth() {
        let frame = EventFrame(
            type: "event",
            event: "health",
            payload: OpenClawProtocol.AnyCodable(["ok": OpenClawProtocol.AnyCodable(true)]),
            seq: 1,
            stateversion: nil)

        let mapped = MacGatewayChatTransport.mapPushToTransportEvent(.event(frame))
        switch mapped {
        case let .health(ok: ok):
            #expect(ok == true)
        default:
            Issue.record("expected .health from health event, got \(String(describing: mapped))")
        }
    }

    @Test func tickEventMapsToTick() {
        let frame = EventFrame(type: "event", event: "tick", payload: nil, seq: 1, stateversion: nil)
        let mapped = MacGatewayChatTransport.mapPushToTransportEvent(.event(frame))
        #expect({
            if case .tick = mapped { return true }
            return false
        }())
    }

    @Test func chatEventMapsToChat() {
        let payload = OpenClawProtocol.AnyCodable([
            "runId": OpenClawProtocol.AnyCodable("run-1"),
            "sessionKey": OpenClawProtocol.AnyCodable("main"),
            "state": OpenClawProtocol.AnyCodable("final"),
        ])
        let frame = EventFrame(type: "event", event: "chat", payload: payload, seq: 1, stateversion: nil)
        let mapped = MacGatewayChatTransport.mapPushToTransportEvent(.event(frame))

        switch mapped {
        case let .chat(chat):
            #expect(chat.runId == "run-1")
            #expect(chat.sessionKey == "main")
            #expect(chat.state == "final")
        default:
            Issue.record("expected .chat from chat event, got \(String(describing: mapped))")
        }
    }

    @Test func execApprovalEventsMapToTransportEvents() {
        let requested = EventFrame(
            type: "event",
            event: "exec.approval.requested",
            payload: OpenClawProtocol.AnyCodable([
                "id": OpenClawProtocol.AnyCodable("approval-1"),
                "createdAtMs": OpenClawProtocol.AnyCodable(123.0),
                "expiresAtMs": OpenClawProtocol.AnyCodable(456.0),
                "request": OpenClawProtocol.AnyCodable([
                    "command": OpenClawProtocol.AnyCodable("npm test"),
                    "kind": OpenClawProtocol.AnyCodable("permission-escalation"),
                    "cwd": OpenClawProtocol.AnyCodable("/tmp/work"),
                    "sessionKey": OpenClawProtocol.AnyCodable("main"),
                ]),
            ]),
            seq: 2,
            stateversion: nil)

        let requestedMapped = MacGatewayChatTransport.mapPushToTransportEvent(.event(requested))
        switch requestedMapped {
        case let .execApprovalRequested(request):
            #expect(request.id == "approval-1")
            #expect(request.command == "npm test")
            #expect(request.kind == "permission-escalation")
            #expect(request.cwd == "/tmp/work")
            #expect(request.sessionKey == "main")
            #expect(request.createdAtMs == 123.0)
            #expect(request.expiresAtMs == 456.0)
        default:
            Issue.record("expected exec approval request, got \(String(describing: requestedMapped))")
        }

        let resolved = EventFrame(
            type: "event",
            event: "exec.approval.resolved",
            payload: OpenClawProtocol.AnyCodable([
                "id": OpenClawProtocol.AnyCodable("approval-1"),
                "decision": OpenClawProtocol.AnyCodable("approve"),
            ]),
            seq: 3,
            stateversion: nil)

        let resolvedMapped = MacGatewayChatTransport.mapPushToTransportEvent(.event(resolved))
        switch resolvedMapped {
        case let .execApprovalResolved(id, decision):
            #expect(id == "approval-1")
            #expect(decision == "approve")
        default:
            Issue.record("expected exec approval resolution, got \(String(describing: resolvedMapped))")
        }
    }

    @Test func unknownEventMapsToNil() {
        let frame = EventFrame(
            type: "event",
            event: "unknown",
            payload: OpenClawProtocol.AnyCodable(["a": OpenClawProtocol.AnyCodable(1)]),
            seq: 1,
            stateversion: nil)
        let mapped = MacGatewayChatTransport.mapPushToTransportEvent(.event(frame))
        #expect(mapped == nil)
    }

    @Test func seqGapMapsToSeqGap() {
        let mapped = MacGatewayChatTransport.mapPushToTransportEvent(.seqGap(expected: 1, received: 9))
        #expect({
            if case .seqGap = mapped { return true }
            return false
        }())
    }
}
