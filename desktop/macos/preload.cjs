const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('marvDesktop', {
  platform: 'macos',
  version: '0.1.0'
});
