file = open("gen_text.txt", "r")
file2 = open("gene_text_2.txt", "w")

new_file = []
for l in file:
    new_l = str(l).replace('<', '')
    new_l = new_l.replace('>', '')
    file2.write(new_l)
    file2.write("\n")
