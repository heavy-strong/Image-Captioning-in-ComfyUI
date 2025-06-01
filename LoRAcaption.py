import glob
import os
from PIL import Image
from PIL import ImageOps
import numpy as np
import torch
import comfy

# cstr 함수 추가 (로깅용)
class cstr:
    def __init__(self, text):
        self.text = text
    
    @property
    def warning(self):
        return self
    
    @property
    def error(self):
        return self
    
    def print(self):
        print(self.text)

class LoRACaptionSave:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
				"namelist": ("STRING", {"forceInput": True}),
                "path": ("STRING", {"forceInput": True}),
                "text": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "prefix": ("STRING", {"default": " "}),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "save_text_file"
    CATEGORY = "LJRE/LORA"

    def save_text_file(self, text, path, namelist, prefix):

        if not os.path.exists(path):
            cstr(f"The path `{path}` doesn't exist! Creating it...").warning.print()
            try:
                os.makedirs(path, exist_ok=True)
            except OSError as e:
                cstr(f"The path `{path}` could not be created! Is there write access?\n{e}").error.print()

        if text.strip() == '':
            cstr(f"There is no text specified to save! Text is empty.").error.print()

        namelistsplit = namelist.splitlines()
        namelistsplit = [i[:-4] for i in namelistsplit]
        
        
        if prefix.endswith(","):
            prefix += " "
        elif not prefix.endswith(", "):
            prefix+= ", "
        
        file_extension = '.txt'
        filename = self.generate_filename(path, namelistsplit, file_extension)
        
        file_path = os.path.join(path, filename)
        self.writeTextFile(file_path, text, prefix)

        return (text, { "ui": { "string": text } } )
        
    def generate_filename(self, path, namelistsplit, extension):
        counter = 1
        filename = f"{namelistsplit[counter-1]}{extension}"
        while os.path.exists(os.path.join(path, filename)):
            counter += 1
            filename = f"{namelistsplit[counter-1]}{extension}"

        return filename

    def writeTextFile(self, file, content, prefix):
        try:
            with open(file, 'w', encoding='utf-8', newline='\n') as f:
                content= prefix + content
                f.write(content)
        except OSError:
            cstr(f"Unable to save file `{file}`").error.print()

def io_file_list(dir='',pattern='*.txt'):
    res=[]
    for filename in glob.glob(os.path.join(dir,pattern)):
        res.append(filename)
    return res

class LoRACaptionLoad:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
         return {
            "required": {
                "path": ("STRING", {"default":""}),			
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "INT")
    RETURN_NAMES = ("Name list", "path", "Image list", "Image count")

    FUNCTION = "captionload"

    CATEGORY = "LJRE/LORA"

    def captionload(self, path, pattern='*.*'):
        # 파일명 리스트 생성
        text = io_file_list(path, pattern)
        text = list(map(os.path.basename, text))
        text = '\n'.join(text)
        
        # 경로 검증
        if not os.path.isdir(path):
            raise FileNotFoundError(f"path '{path}' cannot be found.")
        
        dir_files = os.listdir(path)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in path '{path}'.")

        # 이미지 파일 필터링
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]
        
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No image files found in path '{path}'.")

        dir_files = [os.path.join(path, x) for x in dir_files]

        images = []
        image_count = 0

        for image_path in dir_files:
            # 디렉토리인지 확인하고 스킵
            if os.path.isdir(image_path):
                continue
            
            # 파일이 존재하는지 확인
            if not os.path.exists(image_path):
                print(f"Warning: File {image_path} not found, skipping...")
                continue
                
            try:
                i = Image.open(image_path)
                i = ImageOps.exif_transpose(i)
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                images.append(image)
                image_count += 1
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

        if len(images) == 0:
            raise ValueError(f"No valid images could be loaded from path '{path}'.")
        
        # 모든 이미지를 하나의 배치로 결합
        if len(images) == 1:
            image_batch = images[0]
        else:
            # 첫 번째 이미지를 기준으로 크기 조정
            image1 = images[0]
            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = comfy.utils.common_upscale(
                        image2.movedim(-1, 1), 
                        image1.shape[2], 
                        image1.shape[1], 
                        "bilinear", 
                        "center"
                    ).movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)
            image_batch = image1

        return text, path, image_batch, len(images)