package ai.marv.android.protocol

import org.junit.Assert.assertEquals
import org.junit.Test

class MarvProtocolConstantsTest {
  @Test
  fun canvasCommandsUseStableStrings() {
    assertEquals("canvas.present", MarvCanvasCommand.Present.rawValue)
    assertEquals("canvas.hide", MarvCanvasCommand.Hide.rawValue)
    assertEquals("canvas.navigate", MarvCanvasCommand.Navigate.rawValue)
    assertEquals("canvas.eval", MarvCanvasCommand.Eval.rawValue)
    assertEquals("canvas.snapshot", MarvCanvasCommand.Snapshot.rawValue)
  }

  @Test
  fun a2uiCommandsUseStableStrings() {
    assertEquals("canvas.a2ui.push", MarvCanvasA2UICommand.Push.rawValue)
    assertEquals("canvas.a2ui.pushJSONL", MarvCanvasA2UICommand.PushJSONL.rawValue)
    assertEquals("canvas.a2ui.reset", MarvCanvasA2UICommand.Reset.rawValue)
  }

  @Test
  fun capabilitiesUseStableStrings() {
    assertEquals("canvas", MarvCapability.Canvas.rawValue)
    assertEquals("camera", MarvCapability.Camera.rawValue)
    assertEquals("screen", MarvCapability.Screen.rawValue)
    assertEquals("voiceWake", MarvCapability.VoiceWake.rawValue)
  }

  @Test
  fun screenCommandsUseStableStrings() {
    assertEquals("screen.record", MarvScreenCommand.Record.rawValue)
  }
}
