This is a repository that uses the Acutis document analysis system--a cocktail of [Surya](https://github.com/VikParuchuri/surya) models combined with [boomb0om](https://github.com/boomb0om/CRAFT-text-detection)'s CRAFT implemetation and [dimdenGD](https://github.com/dimdenGD/chrome-lens-ocr/tree/main)'s Google Lens OCR--to create a large document analysis dataset. 

For a bird's eye view of the documents that were selected for the dataset, take a look at Archive.org's page for [advanced searches](https://archive.org/advancedsearch.php). In the "Mediatype" section, keep the "is" selector from the dropdown menu, and select the "texts" mediatype. Once at the page listing all of Archive.org's "text" mediatypes, we should see that there are tens of millions of such items. On the left sidebar, we can restrict the list: we can show text items of a particular language, creation date, or collection, among other restrictions.

In fact, if we return to the [advanced search](https://archive.org/advancedsearch.php) page, we can receive the results of these filtered searched in CSV format. Navigating further down the page to the "Advanced Search returning JSON, XML, and more" section, we can keep the only returning item under the "Fields to return" to be "identifier": the identifier is the only piece of information that we need for each item. In the "Query:" field, we can enter "mediatype:(texts)" to replicate the exact aforementioned search; if we edit the "Number of results:" field to match the number of identifiers that were found in the first search, and we make sure to select the "CSV format:" box, then in theory, we can retrieve the tens of millions of documents that we first searched, saving it into your downloads folder.

It turns out that in the "Query:" section, we can set conjunctions to add restrictions, as we did with the left sidebar in the first search. In fact, the "identifiers.txt" file of this repo--which is used by our document analysis script to create our dataset--was retrieved from the query:
```
mediatype:(texts) AND language:(Handwritten English)
```
The only changes that we applied to this file was removing the quotation marks in a text editor, and changing the file name from `search.csv` to `identifiers.txt`. On linux, we can also randomize the order of the identifers--which is useful for our purposes--by the command:
```bash
shuf identifiers.txt -o identifiers.txt
```
Now, we have a general idea of how documents of a certain type can be obtained for our conversion into a document analysis dataset. With this out of the way, let's discuss how we can install and run the project. As prerequisites, you'll need to be running on Linux, have CUDA>=12.1, and have conda. Let's start with the installation process.
-----
1. Clone and enter the repository.
```cmd
git clone https://github.com/eaucoin/acutis_data.git
cd acutis_data
```
2. Install the node packages.
```cmd
npm install
```
3. Create the conda environment, `acutis_data`, for the project.
```bash
conda create -n acutis_data python=3.10 -y 
```
4. Activate the conda environment `acutis_data`.
```bash
conda activate acutis_data
```
5. With the environment activated, install the python requirements of the project.
```bash
pip install -r requirements.txt
```
6. In your first few runs, make sure to have `./projectpackages/surya/settings.py` open in a text editor or IDE; you will need to adjust your setting to fit your VRAM resources. In that file, you will need to monitor and change `DETECTOR_BATCH_SIZE` and `ORDER_BATCH_SIZE`, which should both be set to the same number and use about 500MB of VRAM per batch unit; for example, the default batch size of 32 should use 16GB of VRAM.  
-----
We'd now like to run the dataset creation script. By this point, you should have the following:
- The `acutis_data` repo and the `acutis_data` conda environment with the required packages.
- The `acutis_data` node packages installed, which should have created the `./node_modules` in the working directory.
- A set of archive.org identifiers that correspond to the set of documents that you'd like to convert into a dataset. You should have saved them into a .csv file and removed the quotation marks wrapping each file, using a text editor. The file should be saved in the acutis_data repo, with the file name `identifiers.txt`.

If you have all of these things, then you're ready to run the dataset creation script:
```cmd
python start.py --input_dir INPUT_DIR --dataset --chunk_size CHUNK_SIZE
```
- `INPUT_DIR` will be the place where the dataset is created.
- `CHUNK_SIZE` is the number of pages that will be processed at a time; scale this number to your VRAM. In my experience, a chunk size uses about 4GB per unit. So, if you have 20GB of VRAM, then I'd recommend a chunk size of 3 or 4.

The script will update `identifiers.txt` to reflect identifiers that don't have eligible documents--that is, if the archive.org identifier had the "text" mediatype but no documents uploaded, or some other disqualifier. It will also keep track of identifiers whose documents have already been converted into a dataset. If the process is ever terminated and the script is run again, then it will use this information to pick back up where we left off.

If identifiers.txt contains `N` rows, then the completed dataset will look as follows:
```
INPUT_DIR/
└──dashboard.txt
└── identifier_1/
    └── document_name_1/
        ├── 1.html
        ├── 1.boxes
        ├── 2.html
        ├── 2.boxes
        └── ...
    └── document_name_2/
        ├── 1.html
        ├── 1.boxes
        ├── 2.html
        ├── 2.boxes
        └── ...
    ...
    └── document_name_k/
        ├── 1.html
        ├── 1.boxes
        ├── 2.html
        ├── 2.boxes
        └── ...
└── identifier_2/
    └── document_name_1/
        ├── 1.html
        ├── 1.boxes
        ├── 2.html
        ├── 2.boxes
        └── ...
    └── document_name_2/
        ├── 1.html
        ├── 1.boxes
        ├── 2.html
        ├── 2.boxes
        └── ...
    ...
    └── document_name_m/
        ├── 1.html
        ├── 1.boxes
        ├── 2.html
        ├── 2.boxes
        └── ...
...
└── identifier_N/
    └── document_name_1/
        ├── 1.html
        ├── 1.boxes
        ├── 2.html
        ├── 2.boxes
        └── ...
    └── document_name_2/
        ├── 1.html
        ├── 1.boxes
        ├── 2.html
        ├── 2.boxes
        └── ...
    ...
    └── document_name_k/
        ├── 1.html
        ├── 1.boxes
        ├── 2.html
        ├── 2.boxes
        └── ...
```
Ineligible identifiers may show as empty identifier folders.

The file `dashboard.txt` will keep track of relevant information related to the dataset as it is being created. It reads, for example:
```
Processing Statistics
====================
Identifiers with processed documents: 5
Total documents processed: 12
Total pages processed: 456
Last updated: 2024-12-22 15:30:45
```
