import Foundation
import ImageIO
import Vision

func fail(_ message: String) -> Never {
  if let data = "\(message)\n".data(using: .utf8) {
    FileHandle.standardError.write(data)
  }
  Foundation.exit(1)
}

guard CommandLine.arguments.count >= 2 else {
  fail("usage: marv-ocr.swift <image-path>")
}

let imagePath = CommandLine.arguments[1]
let imageUrl = URL(fileURLWithPath: imagePath)

guard let source = CGImageSourceCreateWithURL(imageUrl as CFURL, nil) else {
  fail("unable to open image: \(imagePath)")
}

guard let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
  fail("unable to decode image: \(imagePath)")
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true

let environmentLanguages = (ProcessInfo.processInfo.environment["MARV_OCR_LANGUAGES"] ?? "")
  .split(separator: ",")
  .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
  .filter { !$0.isEmpty }
if !environmentLanguages.isEmpty {
  request.recognitionLanguages = environmentLanguages
}

do {
  let handler = VNImageRequestHandler(cgImage: image, options: [:])
  try handler.perform([request])
  let text = (request.results ?? [])
    .compactMap { observation in observation.topCandidates(1).first?.string }
    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    .filter { !$0.isEmpty }
    .joined(separator: "\n")
  if !text.isEmpty {
    print(text)
  }
} catch {
  fail("ocr failed: \(error.localizedDescription)")
}
