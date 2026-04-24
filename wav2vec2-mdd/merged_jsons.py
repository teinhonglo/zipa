import json

# 讀取 JSON 檔案
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 合併兩個 JSON 字典
def merge_json(dict1, dict2):
    merged_dict = dict1.copy()
    for key, values in dict2.items():
        if key in merged_dict:
            merged_dict[key] += values
        else:
            print(f"key {key} is missed. Continue.")

    # 根據 'id' 去重
    unique_indices = {v: i for i, v in enumerate(merged_dict['id'])}.values()
    unique_dict = {key: [values[i] for i in unique_indices] for key, values in merged_dict.items()}

    return unique_dict

# 寫入合併後的 JSON 檔案
def write_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)

# 主程序
def main():
    file1_path = 'data-json/l2arctic/train.json'
    file2_path = 'data-json/l2arctic/train_l2_unsup.json'
    output_file_path = 'data-json/l2arctic/train_unsup.json'

    json1 = read_json(file1_path)
    json2 = read_json(file2_path)

    merged_json = merge_json(json1, json2)

    write_json(output_file_path, merged_json)

if __name__ == '__main__':
    main()
