import os
import csv
import re
import pandas as pd
import PyPDF2
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found at path: {file_path}")

            with open(file_path, "rb") as pdf_file:
                try:
                    reader = PyPDF2.PdfReader(pdf_file)
                    text = ""

                    for page in reader.pages:
                        text += page.extract_text()

                except PyPDF2.PdfReadError as e:
                    raise PyPDF2.PdfReadError(f"Failed to read PDF file: {str(e)}")

            if not text.strip():
                print(f"Warning: No text content extracted from PDF at {file_path}")
                return None

            return self._extract_dividend_data(text)

        except Exception as e:
            print(f"Error parsing PDF file {file_path}: {str(e)}")
            return None

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

    def _extract_dividend_data(self, raw_data: str) -> list:
        """
        Final robust dividend extraction accounting for financial notation variations.
        Returns structured data: [{period, amount, currency, units}]
        """
        dividend_data = []

        try:
            # Normalize text for consistent processing
            text = re.sub(r'\s+', ' ', raw_data)  # Collapse whitespace
            text = text.replace(u'\xa0', ' ')  # Handle non-breaking spaces

            # 1. Locate Cash Flows section using flexible matching
            cf_section = re.search(
                r'(CONSOLIDATED STATEMENTS? OF CASH FLOWS?.*?)'
                r'(?=CONSOLIDATED STATEMENT|CONSOLIDATED BALANCE|$)', 
                text, 
                re.DOTALL | re.IGNORECASE
            )

            if not cf_section:
                logging.error("Cash flows section not found")
                return []

            cf_content = cf_section.group(1)

            # 2. Extract reporting periods from header
            period_dates = re.findall(
                r'(?:Three|Twelve) Months Ended\s*([A-Z][a-z]+ \d{1,2}, \d{4})',
                cf_content
            )

            if len(period_dates) < 2:
                logging.error(f"Insufficient periods found. Found: {period_dates}")
                return []

            # 3. Find dividend payments with multiple pattern variations
            dividend_pattern = r'''
                (Payments?\ (?:for|of)\ dividends?  # Base pattern
                (?:\ and\ dividend\ equivalents?)?) # Optional suffix
                \D* # Any non-digit separator
                (\({0,1}[\d,]+\){0,1}) # First value (with optional parentheses)
                \D+ # Value separator
                (\({0,1}[\d,]+\){0,1}) # Second value
            '''

            match = re.search(
                dividend_pattern, 
                cf_content, 
                re.VERBOSE | re.IGNORECASE
            )

            if not match:
                # Try alternative patterns
                match = re.search(
                    r'(Dividends?\ paid)\D*([\(\d,]+)\D+([\(\d,]+)', 
                    cf_content, 
                    re.IGNORECASE
                )

            if match:
                logging.debug(f"Matched dividend line: {match.group(0)}")
                # Process both values
                for i, value in enumerate(match.groups()[1:]):
                    clean = value.strip('()').replace(',', '')
                    
                    try:
                        amount = -abs(float(clean)) if '(' in value else float(clean)
                        dividend_data.append({
                            'period': period_dates[i],
                            'amount': amount,
                            'currency': 'USD',
                            'units': 'millions'
                        })
                    except ValueError as e:
                        logging.error(f"Value conversion failed: {value} - {str(e)}")
            else:
                logging.error("All dividend patterns failed to match")
                logging.debug(f"Cash flows content:\n{cf_content}")

        except Exception as e:
            logging.error(f"Critical extraction error: {str(e)}")
            return []

        return dividend_data
