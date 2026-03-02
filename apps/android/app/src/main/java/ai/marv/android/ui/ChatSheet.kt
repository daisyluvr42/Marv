package ai.marv.android.ui

import androidx.compose.runtime.Composable
import ai.marv.android.MainViewModel
import ai.marv.android.ui.chat.ChatSheetContent

@Composable
fun ChatSheet(viewModel: MainViewModel) {
  ChatSheetContent(viewModel = viewModel)
}
