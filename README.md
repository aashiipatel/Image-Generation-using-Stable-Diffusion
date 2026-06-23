# Image Generation using Stable Diffusion

A text-to-image generation application built using Stable Diffusion, Hugging Face Diffusers, PyTorch, and Streamlit. This project transforms natural language prompts into high-quality AI-generated images through a clean and interactive user interface.

---

## Features

* Generate images from text prompts using Stable Diffusion
* Professional Streamlit-based user interface
* Customizable generation parameters

  * Inference Steps
  * Guidance Scale
  * Image Resolution
  * Random Seed
* Download generated images
* Example prompt suggestions
* Real-time generation metrics
* GPU acceleration with CUDA support
* Optimized negative prompting for improved image quality

---

## Project Architecture

```text
User Prompt
      │
      ▼
 Streamlit UI
      │
      ▼
Stable Diffusion Pipeline
      │
      ▼
Image Generation
      │
      ▼
Generated Output
      │
      ▼
Download & Visualization
```

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Diffusers
* Transformers
* Streamlit
* PIL (Python Imaging Library)
* CUDA (GPU Acceleration)

---

## Project Structure

```text
Image-Generation-using-Stable-Diffusion/
│
├── app.py
├── 01_StableDiffusion_text2image.ipynb
├── requirements.txt
├── README.md
│
├── generated_images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
│
└── assets/
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Image-Generation-using-Stable-Diffusion.git

cd Image-Generation-using-Stable-Diffusion
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## Example Prompts

```text
A futuristic smart home at sunset, cinematic lighting

A luxury living room with automated lighting, ultra realistic

A cyberpunk city in the rain with neon reflections

A magical floating library above the clouds

A modern glass villa overlooking the ocean
```

---

## Adjustable Parameters

| Parameter      | Description                                |
| -------------- | ------------------------------------------ |
| Steps          | Controls image quality and generation time |
| Guidance Scale | Controls prompt adherence                  |
| Resolution     | Output image dimensions                    |
| Seed           | Reproduce image generations                |

---

## Sample Output

Add generated image screenshots here:

```markdown
![Sample Output](assets/sample.png)
```

---

## Hardware Requirements

### Recommended

* NVIDIA GPU with CUDA support
* 8GB+ VRAM

### Tested On

* Google Colab T4 GPU
* NVIDIA T4 (16GB VRAM)

---

## Future Improvements

* Multiple image generation
* Image-to-Image generation
* Inpainting support
* Style presets
* Prompt enhancement using LLMs
* Model selection interface
* Gallery management

---

## Learning Outcomes

This project demonstrates:

* Generative AI fundamentals
* Diffusion Models
* Stable Diffusion architecture
* Prompt Engineering
* GPU-accelerated inference
* Streamlit application development
* Hugging Face ecosystem integration

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Submit a Pull Request

---

## License

This project is intended for educational and research purposes.

---

## Author

**Aashi Patel**

Built as part of a Generative AI and Computer Vision learning project using Stable Diffusion and Streamlit.
