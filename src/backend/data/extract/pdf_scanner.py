import os
import csv
import pandas as pd
import PyPDF2

class FileUploadParser:
    """Class for uploading and parsing pdfs, csvs & .xls files"""
    
    def __init__(self, upload_dir="uploads"):
        # Directory where files will be saved
        self.upload_dir = upload_dir
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
    
    def upload_file(self, file: any, filename: str) -> str:
        """
        Saves the uploaded file to the specified directory.
        """
        file_path = os.path.join(self.upload_dir, filename)
        with open(file_path, "wb") as f:
            f.write(file.read())
        return file_path
    
    def parse_file(self, file_path: str) -> None:
        """
        Detects the file type and routes to the appropriate parser.
        """
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        if file_extension == ".pdf":
            return self._parse_pdf(file_path)
        elif file_extension == ".csv":
            return self._parse_csv(file_path)
        elif file_extension in [".xls", ".xlsx"]:
            return self._parse_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _parse_pdf(self, file_path: str) -> None:
        """
        Parses PDF files for dividend company data.
        """
        with open(file_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        
        # Custom logic to extract dividend data from the text
        return self._extract_dividend_data(text)
    
    def _parse_csv(self, file_path: str) -> None:
        """
        Parses CSV files for dividend company data.
        """
        data = []
        with open(file_path, mode="r", encoding="utf-8") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                data.append(row)
        
        # Custom logic to process rows for dividend data
        return self._extract_dividend_data(data)
    
    def _parse_excel(self, file_path: str) -> None:
        """
        Parses Excel files for dividend company data.
        """
        df = pd.read_excel(file_path)
        data = df.to_dict(orient="records")
        
        # Custom logic to process rows for dividend data
        return self._extract_dividend_data(data)
    
    def _extract_dividend_data(self, raw_data: any) -> None:
        """
        Custom logic for extracting dividend company data.
        """
        # extraction logic here