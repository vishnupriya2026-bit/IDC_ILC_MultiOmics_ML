import os
import requests
import pandas as pd

def get_gdc_metadata(data_dir):
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return pd.DataFrame()

    # Extract UUID subdirectories
    file_uuids = [f for f in os.listdir(data_dir) if len(f) == 36]
    print(f"Found {len(file_uuids)} sample UUID folders in {data_dir}")

    if not file_uuids:
        return pd.DataFrame()

    url = "https://api.gdc.cancer.gov/files"
    payload = {
        "filters": {"op": "in", "content": {"field": "file_id", "value": file_uuids}},
        "fields": "file_id,file_name,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type",
        "format": "JSON",
        "size": len(file_uuids)
    }

    response = requests.post(url, json=payload).json()
    records = []

    for hit in response["data"]["hits"]:
        case = hit["cases"][0]
        sample = case["samples"][0]
        full_barcode = sample["submitter_id"]
        records.append({
            "file_id": hit["file_id"],
            "patient_id": case["submitter_id"],
            "sample_barcode": full_barcode,
            "sample_type": sample["sample_type"],
            "sample_code": full_barcode.split("-")[3][:2]
        })

    return pd.DataFrame(records)

base_path = os.path.expanduser("~/phd_research/data")

print("Processing RNA-seq metadata...")
rna_df = get_gdc_metadata(os.path.join(base_path, "rna_seq"))
rna_df.to_csv(os.path.join(base_path, "rna_seq_metadata.csv"), index=False)

print("Processing RPPA metadata...")
rppa_df = get_gdc_metadata(os.path.join(base_path, "rppa"))
rppa_df.to_csv(os.path.join(base_path, "rppa_metadata.csv"), index=False)

print("Processing Methylation metadata...")
meth_df = get_gdc_metadata(os.path.join(base_path, "methylation"))
meth_df.to_csv(os.path.join(base_path, "methylation_metadata.csv"), index=False)

print("Metadata extraction complete.")
