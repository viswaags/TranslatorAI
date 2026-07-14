const state = {
  activeTab: 'image',
  langs: {
    image: { src: 'auto', tgt: 'eng_Latn' },
    speech: { src: 'auto', tgt: 'eng_Latn' },
    text: { src: 'auto', tgt: 'eng_Latn' },
  },
  recording: false,
  mediaRecorder: null,
  audioChunks: [],
  selectedAudioFile: null,
  processingStart: null,
  cameraStream: null,
  facingMode: 'environment',
  torchOn: false,
  originalImageBlob: null,
  enhancedImageBlob: null,
};
