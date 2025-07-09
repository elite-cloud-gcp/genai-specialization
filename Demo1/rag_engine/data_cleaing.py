import csv
import json
import ast
import re

# 输入输出文件路径
input_csv = 'input.txt'  # 假设你的csv内容保存在input.txt
output_jsonl_prefix = 'output_'  # 输出jsonl文件前缀

# 解析 product_specifications 字段

def parse_product_specifications(spec_str):
    # 处理特殊的=>格式为标准json
    spec_str = spec_str.replace('=>', ':')
    try:
        spec_dict = json.loads(spec_str)
    except Exception:
        try:
            spec_dict = ast.literal_eval(spec_str)
        except Exception:
            return ''
    if isinstance(spec_dict, dict):
        spec_list = spec_dict.get('product_specification', [])
    else:
        spec_list = spec_dict
    spec_texts = []
    for item in spec_list:
        if isinstance(item, dict):
            key = item.get('key')
            value = item.get('value')
            if key and value:
                spec_texts.append(f"{key}: {value}")
            elif value:
                spec_texts.append(f"{value}")
    return ', '.join(spec_texts)

# 解析 product_category_tree 字段

def clean_category_tree(cat_str):
    try:
        # 去除外层[]和引号
        cat_str = ast.literal_eval(cat_str)[0]
    except Exception:
        pass
    cat_str = cat_str.replace('>>', ',')
    cat_str = cat_str.replace('  ', ' ')
    cat_str = cat_str.replace("'", '').replace('"', '')
    cat_str = cat_str.strip()
    # 只保留前5级
    cats = [c.strip() for c in cat_str.split(',')]
    return ', '.join(cats[:5])

# 解析 image 字段

def parse_image_urls(image_str):
    try:
        urls = ast.literal_eval(image_str)
        if isinstance(urls, str):
            urls = [urls]
        return urls
    except Exception:
        return []

def process_row(row):
    # id
    uniq_id = row['uniq_id']
    # product_name
    product_name = row['product_name']
    # brand
    brand = row['brand']
    # category
    category = clean_category_tree(row['product_category_tree'])
    # description
    description = row['description']
    # product_specifications
    spec_text = parse_product_specifications(row['product_specifications'])
    # text_for_embedding
    text_parts = [
        f"{product_name}.",
        f"Brand: {brand}.",
        f"Category: {category}."
    ]
    # 融合描述和规格
    desc_spec = ''
    if description:
        desc_spec += description.strip()
    if spec_text:
        if desc_spec:
            desc_spec += ' '
        desc_spec += spec_text
    if desc_spec:
        text_parts.append(desc_spec)
    text_for_embedding = ' '.join(text_parts)
    # metadata
    metadata = {
        'product_name': product_name,
        'brand': brand,
        'retail_price': int(row['retail_price']) if row['retail_price'].isdigit() else row['retail_price'],
        'discounted_price': int(row['discounted_price']) if row['discounted_price'].isdigit() else row['discounted_price'],
        'image_urls': parse_image_urls(row['image']),
        'product_url': row['product_url']
    }
    return {
        'id': uniq_id,
        'text_for_embedding': text_for_embedding,
        'metadata': metadata
    }

def main():
    with open(input_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        batch = []
        file_idx = 1
        for i, row in enumerate(reader, 1):
            item = process_row(row)
            batch.append(item)
            if i % 5000 == 0:
                with open(f'{output_jsonl_prefix}{file_idx}.jsonl', 'w', encoding='utf-8') as out_f:
                    for obj in batch:
                        out_f.write(json.dumps(obj, ensure_ascii=False) + '\n')
                batch = []
                file_idx += 1
        # 写最后一批
        if batch:
            with open(f'{output_jsonl_prefix}{file_idx}.jsonl', 'w', encoding='utf-8') as out_f:
                for obj in batch:
                    out_f.write(json.dumps(obj, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
