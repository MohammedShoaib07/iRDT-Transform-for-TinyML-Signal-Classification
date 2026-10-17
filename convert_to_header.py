# convert_to_header.py
# Run once on PC after downloading irdt_ecg_model.tflite from Colab.
# Produces irdt_arduino/irdt_model_data.h for the Arduino sketch.

import os, struct

TFLITE_PATH = os.path.join(os.path.dirname(__file__), 'irdt_ecg_model.tflite')
OUT_DIR     = os.path.join(os.path.dirname(__file__), 'irdt_arduino')
OUT_PATH    = os.path.join(OUT_DIR, 'irdt_model_data.h')

os.makedirs(OUT_DIR, exist_ok=True)

with open(TFLITE_PATH, 'rb') as f:
    data = f.read()

hex_vals = ', '.join(f'0x{b:02x}' for b in data)

with open(OUT_PATH, 'w') as f:
    f.write('#pragma once\n')
    f.write(f'// Auto-generated from {os.path.basename(TFLITE_PATH)}\n')
    f.write(f'// Model size: {len(data)} bytes ({len(data)/1024:.1f} KB)\n\n')
    f.write('alignas(8) const unsigned char irdt_model_data[] = {\n')
    # write 12 bytes per line for readability
    for i in range(0, len(data), 12):
        chunk = ', '.join(f'0x{b:02x}' for b in data[i:i+12])
        f.write(f'  {chunk},\n')
    f.write('};\n')
    f.write(f'const unsigned int irdt_model_data_len = {len(data)};\n')

print(f"Written {OUT_PATH}")
print(f"Model size: {len(data)} bytes ({len(data)/1024:.1f} KB)")
