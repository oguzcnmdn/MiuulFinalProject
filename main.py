import numpy as np
import pandas as pd
import requests


def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"{filename} downloaded successfully.")
    else:
        print(f"Failed to download {filename}.")

file_info = [
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/287e7fc9-6d92-4019-ac58-ff6bca6e6151/download/traffic_density_202207.csv", "traffic_density_202207.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/acd85951-6d23-4b50-bac6-d941f92af1ad/download/traffic_density_202208.csv", "traffic_density_202208.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/a5da03fe-4a89-493b-ae60-aeb132511be9/download/traffic_density_202209.csv", "traffic_density_202209.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/72183a60-d47f-4dc9-b1dc-fced0649dcf5/download/traffic_density_202210.csv", "traffic_density_202210.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/7f463362-a580-41d9-a86a-a542818e7542/download/traffic_density_202211.csv", "traffic_density_202211.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/dc788908-2b75-434f-9f3f-ef82ff33a158/download/traffic_density_202212.csv", "traffic_density_202212.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/42fa7a5f-29f1-4b38-9dfa-ac7c8fe3c77d/download/traffic_density_202301.csv", "traffic_density_202301.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/366befd8-defd-4f79-a3d2-0e7948c649ff/download/traffic_density_202302.csv", "traffic_density_202302.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/6a60b03a-bf25-4575-9dce-e21fe0e04e77/download/traffic_density_202303.csv", "traffic_density_202303.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/ce65562e-0d17-4d7e-8090-9484990a8f2b/download/traffic_density_202304.csv", "traffic_density_202304.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/d0a71c11-47d2-4f98-8745-c9446b10bf18/download/traffic_density_202305.csv", "traffic_density_202305.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/a99913df-dccc-4b7d-b6e3-963ccb5d27b1/download/traffic_density_202306.csv", "traffic_density_202306.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/3de18c1e-57c0-4493-9b75-5a896edae0ff/download/traffic_density_202307.csv", "traffic_density_202307.csv")
]

for url, filename in file_info:
    download_file(url, filename)


def concatenate_csv_files(filenames, output_filename):
    dfs = []

    for filename in filenames:
        df = pd.read_csv(filename)
        dfs.append(df)

    concatenated_df = pd.concat(dfs, ignore_index=True)

    concatenated_df.to_csv(output_filename, index=False)
    print(f"Concatenated data saved to {output_filename}.")


input_filenames = [filename for url, filename in file_info]

output_filename = "concatenated_traffic_data.csv"

concatenate_csv_files(input_filenames, output_filename)


