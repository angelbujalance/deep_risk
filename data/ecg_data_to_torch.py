import os
import pandas as pd
from lxml import etree
import xml.etree.ElementTree as ET
import torch

initial_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(initial_file_dir)))

print(initial_file_dir)
print(parent_dir)

os.chdir(parent_dir)

# Make directory the path with the ECG data
ecg_folder_path = "/gpfs/work2/0/aus20644/data/ukbiobank/ecg/20205_12_lead_ecg_at_rest/imaging_visit_array_0"
leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
         'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
tensors = []

def extract_waveform_data(lead):
    # Find the <WaveformData> element with the specific lead
    waveform_element = root.find(f"StripData/WaveformData[@lead='{lead}']")
    if waveform_element is not None:
        # Extract the text content (comma-separated values)
        waveform_text = waveform_element.text.strip()
        # Convert the string to a list of floats
        waveform_values = [float(x) for x in waveform_text.split(',')]
        assert len(waveform_values) == 5000
        return waveform_values
    return print(f"waveform not found in {file_path} for lead {lead}")

# List all files in the folder
for filename in os.listdir(ecg_folder_path):
    # Check if the file has a .xml extension
    waveforms_file = []
    if filename.endswith('.xml'):
        file_path = os.path.join(ecg_folder_path, filename)
        
        # Parse the XML file
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Get all the 'lead' attributes from the <WaveformData> elements
            for lead in leads:
                waveform_values = extract_waveform_data(lead)
                waveforms_file.append(waveform_values)

            file_tensor = torch.tensor(waveforms_file)
            assert file_tensor.shape == torch.Size([12, 5000])
            tensors.append(file_tensor)

            # Print the root element or any specific content
            #print(f"Content of {filename}:")
            #ET.dump(root)  # This prints the entire XML structure
            #print("\n" + "="*40 + "\n")
        except ET.ParseError as e:
            print(f"Error parsing {filename}: {e}")


combined_tensor = torch.stack(tensors)

torch.save(combined_tensor, "full_data.pt")