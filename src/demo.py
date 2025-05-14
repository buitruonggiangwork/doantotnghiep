import torch
import torchaudio
import tkinter as tk
from tkinter import filedialog, messagebox
from model_driver import LSTMDriver

driver = LSTMDriver()
driver._load_checkpoint()

# Hàm dự đoán âm thanh
def predict_label_with_confidence(file_path):
    waveform, sr = torchaudio.load(file_path)
    waveform = waveform[0]  # lấy kênh đầu tiên nếu là stereo

    target_length = 64000
    if waveform.size(0) < target_length:
        pad = target_length - waveform.size(0)
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:target_length]

    waveform = waveform.unsqueeze(0).to(driver.device)

    driver.model.eval()
    with torch.no_grad():
        output = driver.model(waveform)
        probabilities = output.squeeze().cpu().numpy()
        label = "Thật" if probabilities[0] > probabilities[1] else "Giả"
        confidence = max(probabilities)
        return label, confidence

# Tạo giao diện
def run_gui():
    window = tk.Tk()
    window.title("Dự đoán âm thanh bằng LSTM")
    window.geometry("400x200")

    file_label = tk.Label(window, text="Chưa chọn file", wraplength=380)
    file_label.pack(pady=10)

    result_label = tk.Label(window, text="", font=("Helvetica", 14))
    result_label.pack(pady=10)

    def choose_file():
        file_path = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if file_path:
            file_label.config(text=f"Đã chọn: {file_path}")
            try:
                label, confidence = predict_label_with_confidence(file_path)
                result_label.config(
                    text=f"Kết quả: {label} ({confidence*100:.2f}%)"
                )
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể dự đoán: {str(e)}")

    choose_button = tk.Button(window, text="Chọn file âm thanh", command=choose_file)
    choose_button.pack(pady=20)

    window.mainloop()

if __name__ == "__main__":
    run_gui()
