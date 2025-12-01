"""
NutriGreen Vision Manager - Optimized for 6GB VRAM
Dynamic loading: Models load on-demand and clear when switching
"""

import torch
import time
import os
import io
import base64
import json
import re
from PIL import Image
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig
)
from openai import OpenAI, APIError

# Load environment variables
load_dotenv()


class VisionManager:
    """
    Dynamic Vision Manager - Optimized for 6GB VRAM

    Models are loaded on-demand and cleared when switching modes.
    This allows working with limited VRAM.
    """

    def __init__(self, openai_api_key=None):
        """
        Initialize VisionManager

        Args:
            openai_api_key: Optional OpenAI API key for premium mode
        """
        self.moondream_model = None
        self.moondream_tokenizer = None
        self.llava_model = None
        self.llava_processor = None
        self.openai_client = None

        # Get API key from parameter or environment
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')

        # Track currently loaded model
        self.current_model = None

        print("🔧 VisionManager initialized (Dynamic Loading Mode)")
        print(f"   Premium Mode: {'✅ Ready' if self.openai_api_key else '❌ No API key'}")
        print("\n💡 Models will load on-demand to save VRAM")

    def clear_all_models(self):
        """Clear all loaded models from VRAM"""
        if self.moondream_model is not None:
            del self.moondream_model
            del self.moondream_tokenizer
            self.moondream_model = None
            self.moondream_tokenizer = None
            print("🧹 Cleared Moondream from VRAM")

        if self.llava_model is not None:
            del self.llava_model
            del self.llava_processor
            self.llava_model = None
            self.llava_processor = None
            print("🧹 Cleared LLaVA from VRAM")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
            print(f"💾 Free VRAM: {free_vram:.2f} GB")

        self.current_model = None

    def load_moondream(self):
        """Load Moondream2 model"""
        if self.moondream_model is not None:
            print("✅ Moondream already loaded")
            return True

        try:
            print("\n📥 Loading Moondream2...")

            # Clear other models first
            if self.current_model == "llava":
                print("   Clearing LLaVA to make room...")
                if self.llava_model is not None:
                    del self.llava_model
                    del self.llava_processor
                    self.llava_model = None
                    self.llava_processor = None
                    torch.cuda.empty_cache()

            self.moondream_model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                revision="2024-08-26",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )

            self.moondream_tokenizer = AutoTokenizer.from_pretrained(
                "vikhyatk/moondream2",
                revision="2024-08-26",
                trust_remote_code=True
            )

            self.current_model = "moondream"

            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated(0) / 1024**3
                print(f"✅ Moondream loaded - VRAM used: {vram_used:.2f} GB")
            else:
                print("✅ Moondream loaded on CPU")

            return True

        except Exception as e:
            print(f"❌ Error loading Moondream: {e}")
            return False

    def load_llava(self):
        """Load LLaVA-1.5 model with 4-bit quantization"""
        if self.llava_model is not None:
            print("✅ LLaVA already loaded")
            return True

        try:
            print("\n📥 Loading LLaVA-1.5 (4-bit quantization)...")

            # Clear other models first
            if self.current_model == "moondream":
                print("   Clearing Moondream to make room...")
                if self.moondream_model is not None:
                    del self.moondream_model
                    del self.moondream_tokenizer
                    self.moondream_model = None
                    self.moondream_tokenizer = None
                    torch.cuda.empty_cache()

            # Check available VRAM
            if torch.cuda.is_available():
                free_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
                print(f"   Free VRAM: {free_vram:.2f} GB")

                if free_vram < 4.0:
                    print("\n⚠️ Not enough free VRAM for LLaVA (needs ~4GB)")
                    print("   Trying with CPU offloading...")

            # 4-bit quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True
            )

            self.llava_processor = LlavaNextProcessor.from_pretrained(
                "llava-hf/llava-v1.6-mistral-7b-hf"
            )

            self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
                "llava-hf/llava-v1.6-mistral-7b-hf",
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            self.current_model = "llava"

            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated(0) / 1024**3
                print(f"✅ LLaVA loaded - VRAM used: {vram_used:.2f} GB")
            else:
                print("✅ LLaVA loaded on CPU")

            return True

        except Exception as e:
            print(f"❌ Error loading LLaVA: {e}")
            import traceback
            traceback.print_exc()
            return False

    def analyze_image(self, image, mode="quick", yolo_detections=None, ocr_results=None):
        """
        Analyze image with selected mode

        Args:
            image: PIL Image or path to image
            mode: 'quick', 'standard', or 'premium'
            yolo_detections: List of YOLO detections
            ocr_results: OCR results

        Returns:
            dict: Analysis results
        """
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image)

        # Build context from YOLO and OCR
        context = self._build_context(yolo_detections, ocr_results)

        # Route to appropriate mode
        if mode == "quick":
            return self._analyze_quick(image, context)
        elif mode == "standard":
            return self._analyze_standard(image, context)
        elif mode == "premium":
            return self._analyze_premium(image, context)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _build_context(self, yolo_detections, ocr_results):
        """Build context from YOLO and OCR results"""
        context = {}

        if yolo_detections:
            labels = [d.get('label', '') for d in yolo_detections]
            context['detected_labels'] = ', '.join(labels)

        if ocr_results:
            context['ocr_text'] = ocr_results.get('raw_text', '')[:500]
            context['brand'] = ocr_results.get('brand')
            context['product_name'] = ocr_results.get('product_name')

        return context

    def _analyze_quick(self, image, context):
        """Quick mode with Moondream2"""
        # Load model if not loaded
        if not self.load_moondream():
            return {"error": "Failed to load Moondream model"}

        try:
            prompt = "Analyze this food product. "
            if context.get('detected_labels'):
                prompt += f"Labels detected: {context['detected_labels']}. "
            prompt += "Provide: category, main ingredients (if visible), and key features."

            start_time = time.time()
            enc_image = self.moondream_model.encode_image(image)
            answer = self.moondream_model.answer_question(enc_image, prompt, self.moondream_tokenizer)
            elapsed = time.time() - start_time

            return {
                "mode": "quick",
                "response": answer,
                "time_seconds": round(elapsed, 2),
                "context_used": context
            }

        except Exception as e:
            return {"error": f"Quick mode error: {str(e)}"}

    def _analyze_standard(self, image, context):
        """Standard mode with LLaVA-1.5"""
        # Load model if not loaded
        if not self.load_llava():
            return {"error": "Failed to load LLaVA model"}

        try:
            prompt = "Analyze this food product in detail. "
            if context.get('detected_labels'):
                prompt += f"Labels detected: {context['detected_labels']}. "
            if context.get('ocr_text'):
                prompt += f"Text visible: {context['ocr_text'][:200]}... "
            prompt += "Provide: category, ingredients, nutritional highlights, and dietary suitability."

            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }]

            prompt_text = self.llava_processor.apply_chat_template(conversation, add_generation_prompt=True)

            start_time = time.time()
            inputs = self.llava_processor(images=image, text=prompt_text, return_tensors="pt")

            # Move to GPU only if available and has space
            if torch.cuda.is_available():
                try:
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                except:
                    print("⚠️ Moving inputs to GPU failed, using CPU")

            with torch.no_grad():
                output = self.llava_model.generate(**inputs, max_new_tokens=500, do_sample=False)

            response = self.llava_processor.decode(output[0], skip_special_tokens=True)

            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()

            elapsed = time.time() - start_time

            return {
                "mode": "standard",
                "response": response,
                "time_seconds": round(elapsed, 2),
                "context_used": context
            }

        except Exception as e:
            import traceback
            return {"error": f"Standard mode error: {str(e)}", "traceback": traceback.format_exc()}

    def _analyze_premium(self, image, context):
        """Premium mode with OpenAI GPT-4o"""
        if not self.openai_api_key:
            return {"error": "Premium mode not available - no API key. Check your .env file."}

        try:
            # Initialize client if needed
            if not self.openai_client:
                self.openai_client = OpenAI(api_key=self.openai_api_key)

            # Encode image
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

            prompt = "Analyze this food product and provide structured information.\n\n"
            if context.get('detected_labels'):
                prompt += f"Labels detected: {context['detected_labels']}\n"
            if context.get('ocr_text'):
                prompt += f"Text visible: {context['ocr_text'][:300]}\n"

            prompt += """\nRespond ONLY with a JSON object (no markdown) with this structure:
{
  "category": "specific food category",
  "product_type": "brief description",
  "description": "2-3 sentence description",
  "key_ingredients": ["list of main ingredients"],
  "usage_suggestions": "how to use this product",
  "suitable_for": ["dietary types"]
}"""

            start_time = time.time()
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "low"
                            }
                        }
                    ]
                }],
                max_tokens=800
            )

            elapsed = time.time() - start_time

            response_text = response.choices[0].message.content
            response_text = re.sub(r'```json\s*|```\s*', '', response_text).strip()

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

            if json_match:
                result = json.loads(json_match.group())
                return {
                    "mode": "premium",
                    "response": result,
                    "time_seconds": round(elapsed, 2),
                    "context_used": context
                }
            else:
                return {"error": "No valid JSON in response", "raw_response": response_text}

        except APIError as e:
            return {"error": f"OpenAI API error: {str(e)}"}
        except Exception as e:
            return {"error": f"Premium mode error: {str(e)}"}

    def get_status(self):
        """Get current status of all modes"""
        return {
            "quick": {
                "available": True,
                "loaded": self.moondream_model is not None
            },
            "standard": {
                "available": True,
                "loaded": self.llava_model is not None
            },
            "premium": {
                "available": self.openai_api_key is not None,
                "loaded": self.openai_client is not None
            },
            "current_model": self.current_model
        }
