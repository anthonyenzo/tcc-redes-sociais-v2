import csv

input_path = r"C:\TCC2\data\perfis_base.csv"
output_path = r"C:\TCC2\data\perfis_base_modified.csv"

with open(input_path, encoding="utf-8") as infile, open(output_path, "w", newline='', encoding="utf-8") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for i, row in enumerate(reader):
        if i == 0:  # cabeçalho - não altera
            writer.writerow(row)
        else:
            writer.writerow(row + ["1.0"])
            
print("Arquivo modificado salvo em:", output_path)