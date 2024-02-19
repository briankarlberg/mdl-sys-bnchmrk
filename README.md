# mdl-sys-bnchmrk

Cancer model systems benchmarking

## Usage

| Argument             | Short | Usage            | Description                                                                                                                                                                                                          | Required           |
|----------------------|-------|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| --epoch              | -e    | -e 50            | The amount of epoch for training the model.                                                                                                                                                                          | No   (Default 50)  |
| --batch_size         | -b    | -b 64            | The batch size being used for training of the model.                                                                                                                                                                 | No   (Default 64)  |
| --latent_space       | -lts  | -lts 1500        | The size of the latent space. Changing this will automatically, change the latent space of the vae as well as the classifier input layers as they have to confirm to the latent space size.                          | No  (Default 1500) |
| --data               | -d    | -d data_file.tsv | The data file to load. Will be split in train and test set. Systems Label as well as Cancer_type label are required                                                                                                  | Yes                |
| --output_dir         | -o    | -o results       | This will create a directory in which all result files will be stored. A subfolder will be created by using the file name. If a folder with the same name already exists an integer extension will be added. e.g. _1 | Yes                |
| --cancer_multiplier  | -cm   | -cm 3.5          | The weight multiplier for the cancer loss                                                                                                                                                                            | No   (Default 2)   |
| --systems_multiplier | -sm   | -sm 2.5          | The weight multiplier for the systems loss                                                                                                                                                                           | No   (Default 2.5) |


### Example

```
cd code
python3 vae.py -o ../results --data ../data/data_file.tsv 
````