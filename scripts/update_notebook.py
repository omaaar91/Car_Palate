import json

with open('OCR.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if 'source' in cell:
        for i in range(len(cell['source'])):
            if 'Tuning_char.pt' in cell['source'][i]:
                cell['source'][i] = cell['source'][i].replace('Tuning_char.pt', 'best_char_afterTuning.pt')
                print('Replaced Tuning_char.pt with best_char_afterTuning.pt')
            if 'batch_results_1_to_9.jpg' in cell['source'][i]:
                cell['source'][i] = cell['source'][i].replace('batch_results_1_to_9.jpg', 'batch_results_1_to_9_after_tuning.jpg')
                print('Updated batch_results_1_to_9.jpg to batch_results_1_to_9_after_tuning.jpg')

with open('OCR.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Done updating OCR.ipynb!')
