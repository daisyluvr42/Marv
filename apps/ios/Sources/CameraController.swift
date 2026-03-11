import AVFoundation
import OpenClawKit
import Foundation

actor CameraController {
    struct CameraDeviceInfo: Codable, Sendable {
        var id: String
        var name: String
        var position: String
        var deviceType: String
    }

    enum CameraError: LocalizedError, Sendable {
        case cameraUnavailable
        case permissionDenied(kind: String)
        case invalidParams(String)
        case captureFailed(String)

        var errorDescription: String? {
            switch self {
            case .cameraUnavailable:
                "Camera unavailable"
            case let .permissionDenied(kind):
                "\(kind) permission denied"
            case let .invalidParams(message):
                message
            case let .captureFailed(message):
                message
            }
        }
    }

    func snap(params: OpenClawCameraSnapParams) async throws -> (
        format: String,
        base64: String,
        width: Int,
        height: Int)
    {
        let facing = params.facing ?? .front
        let format = params.format ?? .jpg
        let maxWidth = params.maxWidth.flatMap { $0 > 0 ? $0 : nil } ?? 1600
        let quality = Self.clampQuality(params.quality)
        let delayMs = max(0, params.delayMs ?? 0)

        try await self.ensureAccess(for: .video)

        let session = AVCaptureSession()
        session.sessionPreset = .photo

        guard let device = Self.pickCamera(facing: facing, deviceId: params.deviceId) else {
            throw CameraError.cameraUnavailable
        }

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else {
            throw CameraError.captureFailed("Failed to add camera input")
        }
        session.addInput(input)

        let output = AVCapturePhotoOutput()
        guard session.canAddOutput(output) else {
            throw CameraError.captureFailed("Failed to add photo output")
        }
        session.addOutput(output)
        output.maxPhotoQualityPrioritization = .quality

        session.startRunning()
        defer { session.stopRunning() }
        await Self.warmUpCaptureSession()
        await Self.sleepDelayMs(delayMs)

        let settings: AVCapturePhotoSettings = {
            if output.availablePhotoCodecTypes.contains(.jpeg) {
                return AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])
            }
            return AVCapturePhotoSettings()
        }()
        settings.photoQualityPrioritization = .quality

        var delegate: PhotoCaptureDelegate?
        let rawData: Data = try await withCheckedThrowingContinuation { continuation in
            let captureDelegate = PhotoCaptureDelegate(continuation)
            delegate = captureDelegate
            output.capturePhoto(with: settings, delegate: captureDelegate)
        }
        withExtendedLifetime(delegate) {}

        let result = try PhotoCapture.transcodeJPEGForGateway(
            rawData: rawData,
            maxWidthPx: maxWidth,
            quality: quality)

        return (
            format: format.rawValue,
            base64: result.data.base64EncodedString(),
            width: result.widthPx,
            height: result.heightPx)
    }

    func listDevices() -> [CameraDeviceInfo] {
        Self.discoverVideoDevices().map { device in
            CameraDeviceInfo(
                id: device.uniqueID,
                name: device.localizedName,
                position: Self.positionLabel(device.position),
                deviceType: device.deviceType.rawValue)
        }
    }

    private func ensureAccess(for mediaType: AVMediaType) async throws {
        let status = AVCaptureDevice.authorizationStatus(for: mediaType)
        switch status {
        case .authorized:
            return
        case .notDetermined:
            let granted = await withCheckedContinuation(isolation: nil) { continuation in
                AVCaptureDevice.requestAccess(for: mediaType) { allowed in
                    continuation.resume(returning: allowed)
                }
            }
            if !granted {
                throw CameraError.permissionDenied(kind: mediaType == .video ? "Camera" : "Microphone")
            }
        case .denied, .restricted:
            throw CameraError.permissionDenied(kind: mediaType == .video ? "Camera" : "Microphone")
        @unknown default:
            throw CameraError.permissionDenied(kind: mediaType == .video ? "Camera" : "Microphone")
        }
    }

    private nonisolated static func pickCamera(
        facing: OpenClawCameraFacing,
        deviceId: String?) -> AVCaptureDevice?
    {
        if let deviceId, !deviceId.isEmpty,
           let matched = Self.discoverVideoDevices().first(where: { $0.uniqueID == deviceId })
        {
            return matched
        }

        let position: AVCaptureDevice.Position = facing == .front ? .front : .back
        if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) {
            return device
        }
        return AVCaptureDevice.default(for: .video)
    }

    private nonisolated static func positionLabel(_ position: AVCaptureDevice.Position) -> String {
        switch position {
        case .front:
            return "front"
        case .back:
            return "back"
        default:
            return "unspecified"
        }
    }

    private nonisolated static func discoverVideoDevices() -> [AVCaptureDevice] {
        let types: [AVCaptureDevice.DeviceType] = [
            .builtInWideAngleCamera,
            .builtInUltraWideCamera,
            .builtInTelephotoCamera,
            .builtInDualCamera,
            .builtInDualWideCamera,
            .builtInTripleCamera,
            .builtInTrueDepthCamera,
            .builtInLiDARDepthCamera,
        ]
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: types,
            mediaType: .video,
            position: .unspecified)
        return discovery.devices
    }

    private nonisolated static func warmUpCaptureSession() async {
        try? await Task.sleep(nanoseconds: 350_000_000)
    }

    private nonisolated static func sleepDelayMs(_ delayMs: Int) async {
        guard delayMs > 0 else { return }
        try? await Task.sleep(nanoseconds: UInt64(delayMs) * 1_000_000)
    }

    private nonisolated static func clampQuality(_ quality: Double?) -> Double {
        let resolved = quality ?? 0.9
        return min(1.0, max(0.1, resolved))
    }
}

private final class PhotoCaptureDelegate: NSObject, AVCapturePhotoCaptureDelegate {
    private let continuation: CheckedContinuation<Data, Error>

    init(_ continuation: CheckedContinuation<Data, Error>) {
        self.continuation = continuation
    }

    func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?)
    {
        if let error {
            self.continuation.resume(throwing: error)
            return
        }
        guard let data = photo.fileDataRepresentation() else {
            self.continuation.resume(throwing: CameraController.CameraError.captureFailed("Missing photo data"))
            return
        }
        self.continuation.resume(returning: data)
    }
}
