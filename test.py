import gradio as gr
import requests
import base64
import json
import time
import os
import subprocess
from datetime import datetime
from PIL import Image
import io
from tkinter import Tk, filedialog
from tqdm import tqdm

# ------------------ Inpainting Setup ------------------

webui_server_url = 'http://127.0.0.1:7860'

out_dir = 'api_out'
out_dir_inpaint = os.path.join(out_dir, 'inpaint')
os.makedirs(out_dir_inpaint, exist_ok=True)

FIXED_inpaint = {
    "checkpoint_model": "cosplaymix_v42.safetensors",
    "negative_prompt": "blurred,overlapped,ugly,disfigured,bad quality",
    "seed": 3528166601,
    "steps": 30,
    "sampler_name": "Euler a",
    "width": 576,
    "height": 768,
    "denoising_strength": 0.5,
    "batch_size": 2,
    "cfg_scale": 7.5,
}

lora_prompts_shirt = {
    "brwnV3-000004.safetensors": "upper body,men,wearing,brown,<lora:brwnV3-000004:1.4>,shirt",
    "greentV2-000002.safetensors": "upper body,men,wearing,green,<lora:greentV2-000002:1.4>,t-shirt",
    "new_model.safetensors": "upper body,men,wearing,green,<lora:new_model:1.4>,t-shirt",
    "test_brown.safetensors": "upper body,men,wearing,brown,<lora:test_brown:10.0>,shirt",
    "Demo.safetensors": "upper body,men,wearing,brown,<lora:Demo:1.4>,shirt"
}

lora_prompts_pants = {
    "blackpants.safetensors": "lower body,men,wearing,black,<lora:blackpants:1.5>,pants"
}

# Inpainting Functions
def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))
    return save_path

def call_inpaint_api(payload):
    url = f"{webui_server_url}/sdapi/v1/img2img"
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    if "images" in result:
        save_path = os.path.join(out_dir_inpaint, f"inpaint-{timestamp()}.png")
        decode_and_save_base64(result["images"][0], save_path)
        return save_path
    return None

# ฟังก์ชัน Inpaint ที่เช็คว่าเลือกรุ่น LoRA หรือไม่
def inpaint_image_with_mask(image, lora_shirt, lora_pants):
    # ถ้าไม่ได้เลือก LoRA ให้เว้นช่องนั้นไปเลย
    prompt = ""
    if lora_shirt != "None":
        prompt += lora_prompts_shirt[lora_shirt]
    if lora_pants != "None":
        if prompt:
            prompt += ", "  # ใส่คอมม่ากั้นถ้าเลือกทั้งเสื้อและกางเกง
        prompt += lora_prompts_pants[lora_pants]

    # ถ้าไม่ได้เลือก LoRA เลย
    if not prompt:
        prompt = "men, realistic, wearing casual clothes"

    payload = {
        "prompt": prompt,
        "negative_prompt": FIXED_inpaint["negative_prompt"],
        "seed": FIXED_inpaint["seed"],
        "steps": FIXED_inpaint["steps"],
        "sampler_name": FIXED_inpaint["sampler_name"],
        "width": FIXED_inpaint["width"],
        "height": FIXED_inpaint["height"],
        "batch_size": FIXED_inpaint["batch_size"],
        "cfg_scale": FIXED_inpaint["cfg_scale"],
        "init_images": [encode_image_to_base64(image['image'])],
        "mask": encode_image_to_base64(image['mask']),
        "denoising_strength": FIXED_inpaint["denoising_strength"],
        "inpainting_fill": 1,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 32,
        "resize_mode": 1
    }
    
    save_path = call_inpaint_api(payload)
    if save_path:
        return Image.open(save_path)
    return None
# สร้าง Dropdown LoRA Model แยกเสื้อกับกางเกง
lora_shirt_models = ["None"] + list(lora_prompts_shirt.keys())
lora_pants_models = ["None"] + list(lora_prompts_pants.keys())

# ------------------ LoRA Training Setup ------------------
UPLOAD_FOLDER = "C:/ProjV.2/kohya_ss/dataset"
INSTANCE_FOLDER = os.path.join(UPLOAD_FOLDER, "images")
CLASS_FOLDER = os.path.join(UPLOAD_FOLDER, "regularization")

# Ensure directories exist
os.makedirs(INSTANCE_FOLDER, exist_ok=True)
os.makedirs(CLASS_FOLDER, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, "trained_models"), exist_ok=True)

# Fixed parameters
FIXED_SETTINGS = {
    "network_module": "networks.lora",
    "resolution": "512",
    "cache_latents": True,
    "gradient_checkpointing": True,
    "fp16": True,
    "no_half_vae": True
}

# Validate dataset structure
def validate_folder(folder):
    if not os.path.exists(folder):
        return f"❌ Error: Folder {folder} does not exist!"

    subfolders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    if len(subfolders) == 0 and len(files) == 0:
        return "❌ Error: No files or subfolders found! Please select a folder with images."

    return "✅ Folder structure looks good!"


# Function to run LoRA training
def run_lora_training(model_name, steps, epochs, learning_rate, batch_size, repeats, instance_folder, class_folder, output_dir):
    if not output_dir or not instance_folder or not class_folder:
        return "Please select all folders!"

    instance_folder = instance_folder.replace("\\", "/")
    class_folder = class_folder.replace("\\", "/")
    output_dir = output_dir.replace("\\", "/")
    os.makedirs(output_dir, exist_ok=True)



    python_exec = "C:/ProjV.2/kohya_ss/venv/Scripts/python.exe"
    train_script = "C:/ProjV.2/kohya_ss/sd-scripts/train_network.py"

    train_command = [
        python_exec,
        train_script,
        "--pretrained_model_name_or_path", "C:/ProjV.2/kohya_ss/models/cosplaymix_v42.safetensors",
        "--max_train_steps", str(steps),
        "--learning_rate", str(learning_rate),
        "--output_name", model_name,
        "--resolution", FIXED_SETTINGS["resolution"],
        "--output_dir", output_dir,
        "--train_data_dir", os.path.dirname(instance_folder),
        "--reg_data_dir", class_folder,
        "--dataset_repeats", str(repeats),
        "--enable_bucket",
        "--fp16" if FIXED_SETTINGS["fp16"] else None,
        "--gradient_checkpointing" if FIXED_SETTINGS["gradient_checkpointing"] else None,
        "--cache_latents" if FIXED_SETTINGS["cache_latents"] else None,
        "--no_half_vae" if FIXED_SETTINGS["no_half_vae"] else None,
        "--network_module", FIXED_SETTINGS["network_module"]
    ]
    train_command = [cmd for cmd in train_command if cmd]
    print("🚀 Training command:", " ".join(train_command))
    try:
        process = subprocess.Popen(
            train_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8"
        )

        for line in iter(process.stdout.readline, ''):
            print(line.strip())


        process.stdout.close()
        process.wait()

        if process.returncode == 0:
            return f"🎉 Training completed successfully: {model_name}"
        else:
            return f"❌ Training failed with return code {process.returncode}"

    except Exception as e:
        return f"🚨 Error running training: {str(e)}"

# Function to browse folders
def browse_folder():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory()
    root.destroy()
    return folder if folder else "Folder not selected"
# ------------------ Gradio UI with Tabs ------------------
with gr.Blocks() as app:
    with gr.Tab("🖌️ Inpainting"):
        gr.Markdown("# 🎯 Inpainting with LoRA Models")

        # Dropdown LoRA แยกเสื้อกับกางเกง (มีตัวเลือก None)
        shirt_dropdown = gr.Dropdown(
            lora_shirt_models,
            label="Select Shirt LoRA Model",
            value="None"
        )
        pants_dropdown = gr.Dropdown(
            lora_pants_models,
            label="Select Pants LoRA Model",
            value="None"
        )

        # ช่องอัปโหลดรูปและวาด mask
        image_input = gr.Image(type="pil", tool="sketch", label="Upload and Mask Image")
        output_image = gr.Image(type="pil")

        # ปุ่มเริ่ม inpainting
        gr.Button("Start Inpainting").click(
            fn=inpaint_image_with_mask,
            inputs=[image_input, shirt_dropdown, pants_dropdown],
            outputs=output_image
        )
    # --- LoRA Training Tab ---
    with gr.Tab("🧥 LoRA Training"):
        gr.Markdown("# LoRA Training App")

        model_name = gr.Textbox(label="Model Name", value="new_model")
        gr.Markdown("**ชื่อโมเดล**: ตั้งชื่อโมเดลใหม่ที่คุณจะฝึก เช่น `shirt_v1`,`brownV1`")

        steps = gr.Number(label="Training Steps", value=1000, interactive=True, precision=0)
        gr.Markdown("**จำนวนรอบการฝึก (Steps)**: ยิ่งมาก โมเดลยิ่งเก่งขึ้น แต่ก็ใช้เวลานานขึ้น<br> "
        "**แนะนำ**<br>"
        "**500-1500**steps สำหรับชุดข้อมูลขนาดกลาง")

        epochs = gr.Number(label="Epochs", value=10, interactive=True, precision=0)
        gr.Markdown("**จำนวนรอบการวนซ้ำ (Epochs)**: จำนวนครั้งที่โมเดลจะดูข้อมูลซ้ำๆ<br> "
        "**แนะนำ**<br>"
        "**5-15** รอบกำลังดี")

        learning_rate = gr.Number(label="Learning Rate", value=1e-4, interactive=True)
        gr.Markdown(
            "**อัตราการเรียนรู้ (Learning Rate)**:<br>"
            "เลขน้อย (เช่น 0.0001) → โมเดลเรียนรู้แบบช้าๆ มีความแม่นยำเพิ่มขึ้น แต่ใช้เวลานาน<br>"
            "เลขมาก (เช่น 0.01) → โมเดลเรียนรู้ไว แต่ความแม่นยำจะน้อยลง<br>"
            "**แนะนำ**<br>"
            "ถ้าชุดข้อมูลเยอะ (>200 รูป) → 0.0001 <br>"
            "ถ้าชุดข้อมูลน้อย (<50 รูป) หรืออยากให้ภาพเนียน → 0.00005 "
        )

        batch_size = gr.Number(label="Batch Size", value=1, interactive=True, precision=0)
        gr.Markdown("**จำนวนรูปต่อรอบ (Batch Size)**:<br>"
            "จำนวนรูปที่ให้โมเดลดูพร้อมกัน ค่าน้อยใช้แรมน้อย<br>"
            "**แนะนำ**<br>"
            "1 สำหรับ ram 4-6GB<br>"
            "2 สำหรับ ram 6-8GB<br>"
            "4 สำหรับ ram 8-12GB<br>"
            "8 สำหรับ ram 12-16GB<br>"
            "16+ สำหรับ ram 16GB+ "
        )

        repeats = gr.Number(label="Repeats", value=10, interactive=True, precision=0)
        gr.Markdown("**จำนวนการใช้ซ้ำ (Repeats)**: ให้โมเดลดูรูปเดิมซ้ำกี่รอบ ยิ่งซ้ำมาก โมเดลจะจำได้แม่นขึ้น<br> "
        "**แนะนำ**<br>"
        "**10-20** รอบ")

        instance_dir_box = gr.Textbox(label="Instance Image", interactive=True)
        gr.Markdown("**Instance Image**: เลือกโฟลเดอร์ที่ใส่ภาพของเสื้อผ้าหรือตัวแบบที่คุณต้องการสร้าง เช่น เสื้อคอปกของคุณในมุมต่างๆ ด้านหน้า ด้านหลัง ด้านข้าง ประมาณ **15-30** รูป<br>"
        "**ข้อบังคับ**<br>"
        "1. ชื่อโฟลเดอร์จะต้องมีตัวเลขบอกจำนวนรูปในโฟลเดอร์ เช่น 18_brownshirt<br>"
        "2. ชื่อไฟล์ภาพต้องมี format ที่เป็นการเรียงจำนวน เช่น brownshirt_0001.jpg, shirt_0002.jpg")
        gr.Button("เลือก Folder Instance Image").click(fn=browse_folder, outputs=instance_dir_box)

        class_dir_box = gr.Textbox(label="Class Image", interactive=True)
        gr.Markdown("**Class Image**: เลือกโฟลเดอร์ที่มีภาพทั่วไปในหมวดหมู่เดียวกัน เช่น 'เสื้อคอปกทั่วไป' ประมาณ **100-150** รูป<br>"
        "**ข้อบังคับ**<br>"
        "1. ชื่อโฟลเดอร์จะต้องมีตัวเลขบอกจำนวนรูปในโฟลเดอร์ เช่น 18_shirt<br>"
        "2. ชื่อไฟล์ภาพต้องมี format ที่เป็นการเรียงจำนวน เช่น shirt_0001.jpg, shirt_0002.jpg")
        gr.Button("เลือก Folder Class Image").click(fn=browse_folder, outputs=class_dir_box)

        output_dir_box = gr.Textbox(label="โฟลเดอร์เก็บผลลัพธ์", interactive=True)
        gr.Markdown("**โฟลเดอร์ผลลัพธ์**: เลือกที่เก็บโมเดลที่ฝึกเสร็จแล้ว")
        gr.Button("เลือกโฟลเดอร์ผลลัพธ์").click(fn=browse_folder, outputs=output_dir_box)

        start_training_btn = gr.Button("Training Start")
        output_text = gr.Textbox(label="ผลลัพธ์การฝึก", interactive=False)

        start_training_btn.click(
            fn=run_lora_training,
            inputs=[model_name, steps, epochs, learning_rate, batch_size, repeats, instance_dir_box, class_dir_box, output_dir_box],
            outputs=output_text,
        )



app.launch()
