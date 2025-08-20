import os
import xml.etree.ElementTree as ET

def save_data_to_xml(matrix, filename, path):
    """"
    Save matrix to xml file
    Parameters:
    - matrix: matrix to save 
    - filename: name of the xml file
    - path: path to save the xml file
    """

    full_path = os.path.join(path, filename)
    root = ET.Element("matrix")
    for row in matrix:
        row_elem = ET.SubElement(root, "row")
        for val in row:
            val_elem = ET.SubElement(row_elem, "val")
            val_elem.text = str(val)
    
    tree = ET.ElementTree(root)
    tree.write(full_path)
    print(f"Matrix has been saved as xml to {full_path}")