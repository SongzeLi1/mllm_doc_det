import ast
import json
import re
from pathlib import Path
from typing import Dict

import json5


class JsonDecoder:
    """JSON 解码器类，负责将字符串转换为 JSON 格式"""

    @staticmethod
    def attempt_decode(s: str):
        try:
            return json5.loads(s), None
        except Exception as ex:
            return None, str(ex)

    @staticmethod
    def decode(raw_string: str) -> Dict:
        """解析可能包含注释的JSON字符串"""
        # 策略0: 移除 JSON 字符串中的特殊标记, 如 <|cls|> 和 <|mask_*|>, *标识一个数字符号
        raw_string = raw_string.replace("<|cls|>", "")
        raw_string = re.sub(r"<\|mask_\d+\|>", "", raw_string)

        # 策略1：移除 Markdown 的 JSON 代码块标记，并调整末尾的特殊内容
        s1 = raw_string.replace("```json\n", "").replace("\n```", "")
        if s1.endswith("..."):
            s1 = s1[:-3] + '"..."]]}'
        if s1.endswith(", "):
            s1 = s1[:-1] + '"..."]]}'
        if s1.startswith('"'):
            s1 = s1[1:]
        if s1.endswith('"'):
            s1 = s1[:-1]
        result, error = JsonDecoder.attempt_decode(s1)
        if result is not None:
            return result

        # 策略2：将 '#' 注释替换为 '//'，让 json5 能正确解析
        s2 = re.sub(r"#", "//", raw_string)
        result, error = JsonDecoder.attempt_decode(s2)
        if result is not None:
            return result

        # 策略3：对注释进行改进，确保每个 '//' 注释后有换行符 (不使用函数)
        s3 = re.sub(r"(//[^\n]*(?!\n))", r"\1\n", s2)
        result, error = JsonDecoder.attempt_decode(s3)
        if result is not None:
            return result

        return {"content": raw_string, "error": error}

    @staticmethod
    def save_json_file(file_path: str, data: Dict, encoding: str = "utf-8"):
        file_path = Path(file_path)
        # 确保父目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding=encoding) as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def load_json_file(file_path, encoding: str = "utf-8") -> Dict:
        file_path = Path(file_path)
        assert file_path.exists(), f"File not found: {file_path}"
        with open(file_path, "r", encoding=encoding) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                return {}

    @staticmethod
    def load_file(file_path):
        json_path = Path(file_path)
        if json_path.suffix == ".jsonl":
            data = {}
            # 一行一行读, 每行利用json解析为dict
            for line in open(json_path, "r", encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.update(json.loads(line))
                except json.JSONDecodeError as e:
                    line = line.replace("'", '"')
                    line = line.replace("\n", "").replace("\r", "")
                    data = ast.literal_eval(line)
                    return data
        else:
            data = JsonDecoder.load_json_file(json_path)
        return data


if __name__ == "__main__":
    # 测试代码
    test_string = '{"result": "fake", "mask": "[[948, 60, 1024, 136],[948, 135, 1024, 204],[948, 203, 1024, 272]]", "reason": "We have identified the following clues, where the high-level anomalies are significant doubts worth attention, and the middle-level and low-level findings are reliable evidence.\n\n# High-Level Semantic Anomalies\n\n1. **Content Contrary to Common Sense**:\n - The presence of a duplicate refridgerator in the same kitchen setting is unusual and uncommon. This anomaly raises suspicion about the image\'s authenticity.\n\n# Middle-Level Visual Defects\n\n1. **Traces of Tampered Region or Boundary**:\n - The boundary of the tampered region shows visible artifacts or unnatural edges. \n\n2. **Lighting Inconsistency**:\n - The lighting and shadows in the tampered region do not match the rest of the image. \n\n3. **Perspective Relationships**:\n - The perspective of the copied refridgerator does not align perfectly with the rest of the scene. \n\n# Low-Level Pixel Statistics\n\n1. **Noise Patterns**: \n - The noise pattern in the tampered region is inconsistent with the surrounding areas. \n\n2. **Color and Textural Differences**:\n - There are slight color and texture differences between the tampered region and its surroundings."}'
    decoder = JsonDecoder()
    result = decoder.decode(test_string)
    print(result)  # 输出: {'key': 'value'}
