import CoreLocation
import Foundation
import MarvKit
import UIKit

protocol CameraServicing: Sendable {
    func listDevices() async -> [CameraController.CameraDeviceInfo]
    func snap(params: MarvCameraSnapParams) async throws -> (format: String, base64: String, width: Int, height: Int)
    func clip(params: MarvCameraClipParams) async throws -> (format: String, base64: String, durationMs: Int, hasAudio: Bool)
}

protocol ScreenRecordingServicing: Sendable {
    func record(
        screenIndex: Int?,
        durationMs: Int?,
        fps: Double?,
        includeAudio: Bool?,
        outPath: String?) async throws -> String
}

@MainActor
protocol LocationServicing: Sendable {
    func authorizationStatus() -> CLAuthorizationStatus
    func accuracyAuthorization() -> CLAccuracyAuthorization
    func ensureAuthorization(mode: MarvLocationMode) async -> CLAuthorizationStatus
    func currentLocation(
        params: MarvLocationGetParams,
        desiredAccuracy: MarvLocationAccuracy,
        maxAgeMs: Int?,
        timeoutMs: Int?) async throws -> CLLocation
    func startLocationUpdates(
        desiredAccuracy: MarvLocationAccuracy,
        significantChangesOnly: Bool) -> AsyncStream<CLLocation>
    func stopLocationUpdates()
    func startMonitoringSignificantLocationChanges(onUpdate: @escaping @Sendable (CLLocation) -> Void)
    func stopMonitoringSignificantLocationChanges()
}

protocol DeviceStatusServicing: Sendable {
    func status() async throws -> MarvDeviceStatusPayload
    func info() -> MarvDeviceInfoPayload
}

protocol PhotosServicing: Sendable {
    func latest(params: MarvPhotosLatestParams) async throws -> MarvPhotosLatestPayload
}

protocol ContactsServicing: Sendable {
    func search(params: MarvContactsSearchParams) async throws -> MarvContactsSearchPayload
    func add(params: MarvContactsAddParams) async throws -> MarvContactsAddPayload
}

protocol CalendarServicing: Sendable {
    func events(params: MarvCalendarEventsParams) async throws -> MarvCalendarEventsPayload
    func add(params: MarvCalendarAddParams) async throws -> MarvCalendarAddPayload
}

protocol RemindersServicing: Sendable {
    func list(params: MarvRemindersListParams) async throws -> MarvRemindersListPayload
    func add(params: MarvRemindersAddParams) async throws -> MarvRemindersAddPayload
}

protocol MotionServicing: Sendable {
    func activities(params: MarvMotionActivityParams) async throws -> MarvMotionActivityPayload
    func pedometer(params: MarvPedometerParams) async throws -> MarvPedometerPayload
}

struct WatchMessagingStatus: Sendable, Equatable {
    var supported: Bool
    var paired: Bool
    var appInstalled: Bool
    var reachable: Bool
    var activationState: String
}

struct WatchNotificationSendResult: Sendable, Equatable {
    var deliveredImmediately: Bool
    var queuedForDelivery: Bool
    var transport: String
}

protocol WatchMessagingServicing: AnyObject, Sendable {
    func status() async -> WatchMessagingStatus
    func sendNotification(
        id: String,
        title: String,
        body: String,
        priority: MarvNotificationPriority?) async throws -> WatchNotificationSendResult
}

extension CameraController: CameraServicing {}
extension ScreenRecordService: ScreenRecordingServicing {}
extension LocationService: LocationServicing {}
