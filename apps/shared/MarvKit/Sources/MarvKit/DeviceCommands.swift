import Foundation

public enum MarvDeviceCommand: String, Codable, Sendable {
    case status = "device.status"
    case info = "device.info"
}

public enum MarvBatteryState: String, Codable, Sendable {
    case unknown
    case unplugged
    case charging
    case full
}

public enum MarvThermalState: String, Codable, Sendable {
    case nominal
    case fair
    case serious
    case critical
}

public enum MarvNetworkPathStatus: String, Codable, Sendable {
    case satisfied
    case unsatisfied
    case requiresConnection
}

public enum MarvNetworkInterfaceType: String, Codable, Sendable {
    case wifi
    case cellular
    case wired
    case other
}

public struct MarvBatteryStatusPayload: Codable, Sendable, Equatable {
    public var level: Double?
    public var state: MarvBatteryState
    public var lowPowerModeEnabled: Bool

    public init(level: Double?, state: MarvBatteryState, lowPowerModeEnabled: Bool) {
        self.level = level
        self.state = state
        self.lowPowerModeEnabled = lowPowerModeEnabled
    }
}

public struct MarvThermalStatusPayload: Codable, Sendable, Equatable {
    public var state: MarvThermalState

    public init(state: MarvThermalState) {
        self.state = state
    }
}

public struct MarvStorageStatusPayload: Codable, Sendable, Equatable {
    public var totalBytes: Int64
    public var freeBytes: Int64
    public var usedBytes: Int64

    public init(totalBytes: Int64, freeBytes: Int64, usedBytes: Int64) {
        self.totalBytes = totalBytes
        self.freeBytes = freeBytes
        self.usedBytes = usedBytes
    }
}

public struct MarvNetworkStatusPayload: Codable, Sendable, Equatable {
    public var status: MarvNetworkPathStatus
    public var isExpensive: Bool
    public var isConstrained: Bool
    public var interfaces: [MarvNetworkInterfaceType]

    public init(
        status: MarvNetworkPathStatus,
        isExpensive: Bool,
        isConstrained: Bool,
        interfaces: [MarvNetworkInterfaceType])
    {
        self.status = status
        self.isExpensive = isExpensive
        self.isConstrained = isConstrained
        self.interfaces = interfaces
    }
}

public struct MarvDeviceStatusPayload: Codable, Sendable, Equatable {
    public var battery: MarvBatteryStatusPayload
    public var thermal: MarvThermalStatusPayload
    public var storage: MarvStorageStatusPayload
    public var network: MarvNetworkStatusPayload
    public var uptimeSeconds: Double

    public init(
        battery: MarvBatteryStatusPayload,
        thermal: MarvThermalStatusPayload,
        storage: MarvStorageStatusPayload,
        network: MarvNetworkStatusPayload,
        uptimeSeconds: Double)
    {
        self.battery = battery
        self.thermal = thermal
        self.storage = storage
        self.network = network
        self.uptimeSeconds = uptimeSeconds
    }
}

public struct MarvDeviceInfoPayload: Codable, Sendable, Equatable {
    public var deviceName: String
    public var modelIdentifier: String
    public var systemName: String
    public var systemVersion: String
    public var appVersion: String
    public var appBuild: String
    public var locale: String

    public init(
        deviceName: String,
        modelIdentifier: String,
        systemName: String,
        systemVersion: String,
        appVersion: String,
        appBuild: String,
        locale: String)
    {
        self.deviceName = deviceName
        self.modelIdentifier = modelIdentifier
        self.systemName = systemName
        self.systemVersion = systemVersion
        self.appVersion = appVersion
        self.appBuild = appBuild
        self.locale = locale
    }
}
