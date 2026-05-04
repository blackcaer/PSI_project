import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import requests
import json
import config


class InferenceClient:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
    
    def infer(self, image_path, model_id):
        url = f"{self.api_url}/{model_id}"
        
        with open(image_path, "rb") as f:
            response = requests.post(
                url,
                files={"file": f},
                params={"api_key": self.api_key}
            )
        
        response.raise_for_status()
        return response.json()


class InferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Signs Detection")
        self.root.geometry("800x600")
        
        self.client = InferenceClient(
            api_url=config.API_URL,
            api_key=config.API_KEY
        )
        self.model_id = config.MODEL_ID
        self.current_image_path = None
        self.photo_reference = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        tk.Button(
            btn_frame,
            text="Select Image",
            command=self._select_image,
            width=15,
            height=2
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            btn_frame,
            text="Run Inference",
            command=self._run_inference,
            width=15,
            height=2
        ).pack(side=tk.LEFT, padx=5)
        
        self.image_label = tk.Label(self.root, text="No image selected")
        self.image_label.pack(pady=10)
        
        result_frame = tk.Frame(self.root)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(result_frame, text="Results:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        scrollbar = tk.Scrollbar(result_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text = tk.Text(
            result_frame,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            height=15
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.result_text.yview)
    
    def _select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self._display_image(file_path)
            self.result_text.delete(1.0, tk.END)
    
    def _display_image(self, image_path):
        image = Image.open(image_path)
        image.thumbnail((400, 400))
        self.photo_reference = ImageTk.PhotoImage(image)
        
        self.image_label.config(image=self.photo_reference, text="")
    
    def _run_inference(self):
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        try:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Running inference...\n\n")
            self.root.update()
            
            result = self.client.infer(
                self.current_image_path,
                model_id=self.model_id
            )
            
            self._display_results(result)
            
        except Exception as e:
            messagebox.showerror("Error", f"Inference failed: {str(e)}")
    
    def _display_results(self, result):
        self.result_text.delete(1.0, tk.END)
        
        if "predictions" in result:
            predictions = result["predictions"]
            self.result_text.insert(tk.END, f"Detected {len(predictions)} object(s)\n\n")
            
            for i, pred in enumerate(predictions, 1):
                self.result_text.insert(tk.END, f"Detection {i}:\n")
                self.result_text.insert(tk.END, f"  Class: {pred.get('class', 'N/A')}\n")
                self.result_text.insert(tk.END, f"  Confidence: {pred.get('confidence', 0):.2%}\n")
                self.result_text.insert(tk.END, f"  Position: x={pred.get('x', 0):.1f}, y={pred.get('y', 0):.1f}\n")
                self.result_text.insert(tk.END, f"  Size: {pred.get('width', 0):.1f}x{pred.get('height', 0):.1f}\n\n")
        
        self.result_text.insert(tk.END, "\n" + "="*50 + "\n")
        self.result_text.insert(tk.END, "Full Response:\n")
        self.result_text.insert(tk.END, json.dumps(result, indent=2))


def main():
    root = tk.Tk()
    app = InferenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()


