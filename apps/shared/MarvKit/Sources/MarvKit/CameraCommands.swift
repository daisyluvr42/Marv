import Foundation

public enum MarvCameraCommand: String, Codable, Sendable {
    case list = "camera.list"
    case snap = "camera.snap"
    case clip = "camera.clip"
}

public enum MarvCameraFacing: String, Codable, Sendable {
    case back
    case front
}

public enum MarvCameraImageFormat: String, Codable, Sendable {
    case jpg
    case jpeg
}

public enum MarvCameraVideoFormat: String, Codable, Sendable {
    case mp4
}

public struct MarvCameraSnapParams: Codable, Sendable, Equatable {
    public var facing: MarvCameraFacing?
    public var maxWidth: Int?
    public var quality: Double?
    public var format: MarvCameraImageFormat?
    public var deviceId: String?
    public var delayMs: Int?

    public init(
        facing: MarvCameraFacing? = nil,
        maxWidth: Int? = nil,
        quality: Double? = nil,
        format: MarvCameraImageFormat? = nil,
        deviceId: String? = nil,
        delayMs: Int? = nil)
    {
        self.facing = facing
        self.maxWidth = maxWidth
        self.quality = quality
        self.format = format
        self.deviceId = deviceId
        self.delayMs = delayMs
    }
}

public struct MarvCameraClipParams: Codable, Sendable, Equatable {
    public var facing: MarvCameraFacing?
    public var durationMs: Int?
    public var includeAudio: Bool?
    public var format: MarvCameraVideoFormat?
    public var deviceId: String?

    public init(
        facing: MarvCameraFacing? = nil,
        durationMs: Int? = nil,
        includeAudio: Bool? = nil,
        format: MarvCameraVideoFormat? = nil,
        deviceId: String? = nil)
    {
        self.facing = facing
        self.durationMs = durationMs
        self.includeAudio = includeAudio
        self.format = format
        self.deviceId = deviceId
    }
}
