import librosa
import numpy as np

y, sr = librosa.load(librosa.util.example_audio_file())
D = np.abs(librosa.stft(y))
D
# array([[2.58028018e-03, 4.32422794e-02, 6.61255598e-01, ...,
# 6.82710262e-04, 2.51654536e-04, 7.23036574e-05],
# [2.49403086e-03, 5.15930466e-02, 6.00107312e-01, ...,
# 3.48026224e-04, 2.35853557e-04, 7.54836728e-05],
# [7.82410789e-04, 1.05394892e-01, 4.37517226e-01, ...,
# 6.29352580e-04, 3.38571583e-04, 8.38094638e-05],
# ...,
# [9.48568513e-08, 4.74725084e-07, 1.50052492e-05, ...,
# 1.85637656e-08, 2.89708542e-08, 5.74304337e-09],
# [1.25165826e-07, 8.58259284e-07, 1.11157215e-05, ...,
# 3.49099771e-08, 3.11740926e-08, 5.29926236e-09],
# [1.70630571e-07, 8.92518756e-07, 1.23656537e-05, ...,
# 5.33256745e-08, 3.33264900e-08, 5.13272980e-09]], dtype=float32)

# Use left-aligned frames, instead of centered frames

D_left = np.abs(librosa.stft(y, center=False))

# Use a shorter hop length

D_short = np.abs(librosa.stft(y, hop_length=64))

# Display a spectrogram

import matplotlib.pyplot as plt
librosa.display.specshow(librosa.amplitude_to_db(D,
                                                 ref=np.max),
                         y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show(10)
