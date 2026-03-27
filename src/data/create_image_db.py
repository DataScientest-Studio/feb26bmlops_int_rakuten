import os
from loaders import load_train_validation_sql

class image_db:
    '''parse the sql/df data and copy the image files to the corresponding directory
    '''
    def __init__(self, db_url, step=0, 
                 image_column='image_file', label_column='prdtypecode', sample=None, 
                 input_folder='data/image_data', output_folder = 'data/image_db',
                 seed=42):
        self.db_url = db_url
        self.step = step
        self.image_column = image_column
        self.label_column = label_column
        self.sample = sample
        self.seed = seed
        self.output_folder = output_folder
        self.input_folder = input_folder

    def add_step(self, step):
        self.step = step

        train_df, val_df = load_train_validation_sql(
            db_url=self.db_url,
            step=self.step,
            text_column=self.image_column,
            label_column=self.label_column,
            sample_number=self.sample,
            seed=self.seed
            )

        # Here you would add code to copy the image files based on the text_column values
        if(train_df.empty or val_df.empty):
            print("No data loaded. Check your database connection and query parameters.")
            return
        
        print(f"Train rows: {len(train_df)}")
        print(f"Validation rows: {len(val_df)}")


        # Example: Copy images to train/ and val/ directories based on the text_column values
        for file_name, label in train_df[[self.image_column, self.label_column]].values:
            # Copy file_name to train/ directory
            # print(f"Copying {file_name} to train/ with label {label}")
            self.add_image(file_name, 'train', label)
            # break # Remove this break after testing the first few rows

        for file_name, label in val_df[[self.image_column, self.label_column]].values:
            self.add_image(file_name, 'val', label)
            # break # Remove this break after testing the first few rows

    def add_image(self, file_name, folder, label):
        self.check_create_folder(label=label, folder_path=os.path.join(self.output_folder, folder))
        input_path = os.path.join(self.input_folder, file_name)
        output_path = os.path.join(self.output_folder, folder, str(label), file_name)

        if not os.path.exists(output_path):
            if os.path.exists(input_path):
                # print(f"Adding image {file_name} with label {label}")
                os.system(f"cp {input_path} {output_path}")
            else:
                print(f"Input file {input_path} does not exist. Skipping.")
                return
        else:
            # print(f" -- Output file {output_path} already exists. Skipping copy.")
            pass

    def check_create_folder(self, label, folder_path):
        # print(f"Checking/creating folder: {folder_path}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        if(label is not None):
            folder_path = os.path.join(folder_path, str(label))
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        else:
            print("No label provided, skipping label-specific folder creation.")
            pass
    

if __name__ == "__main__":
    _DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5432/dst_db"
    im_db = image_db(
        db_url=_DEFAULT_DB_URL,
        step=11,
        image_column='image_file',
        label_column='prdtypecode',
        sample=0.8,
        seed=42,
    )

    im_db.add_step(step=1)