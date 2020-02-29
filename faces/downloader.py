import os
import subprocess

def bash(command):
    return subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read().decode()

female_count, male_count, errors = 0, 0, 0
names = []

for i in range(25):
    #os.system("curl -s -L 'https://www.famousbirthdays.com/random' -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:65.0) Gecko/20100101 Firefox/65.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Accept-Language: pl,en-US;q=0.7,en;q=0.3' --compressed -H 'Referer: https://www.famousbirthdays.com/people/onision.html' -H 'content-type: application/x-www-form-urlencoded' -H 'Origin: https://www.famousbirthdays.com' -H 'Connection: keep-alive' -H 'Cookie: __cfduid=dcea8da466e7cb32edd80c67cac2d0b661549570149; __vrz=1.13.4; _ga=GA1.2.1898991895.1549570151; _gid=GA1.2.1853790107.1549731504' -H 'TE: Trailers' --data '' > temp.txt")

    os.system("curl -s -L 'https://www.famousbirthdays.com/search' -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:65.0) Gecko/20100101 Firefox/65.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Accept-Language: pl,en-US;q=0.7,en;q=0.3' --compressed -H 'Referer: https://www.famousbirthdays.com/people/jacob-sartorius.html' -H 'content-type: application/x-www-form-urlencoded' -H 'Origin: https://www.famousbirthdays.com' -H 'Connection: keep-alive' -H 'Cookie: __cfduid=dcea8da466e7cb32edd80c67cac2d0b661549570149; __vrz=1.13.4; _ga=GA1.2.1898991895.1549570151; _gid=GA1.2.1853790107.1549731504; lookup=%231' -H 'TE: Trailers' --data 'q=%23"+str(i)+"' > temp.txt")
    
    print("number of iterations: " + str(i) + " names size: " + str(len(names)) + " errors: " + str(errors))

    num_of_link_line = bash("awk '/img-responsive/{ print NR; exit }' temp.txt")
    if type(num_of_link_line) is not str or len(num_of_link_line) <= 0 or num_of_link_line == "0":
        errors+=1
        continue 
    num_of_link_line = int(num_of_link_line)
    
    y = bash("sed -n '" + str(num_of_link_line) + "p' < temp.txt")
    
    link = ""
    read = 0
    name = ""
    for char in y:
        if char == '"':
            read += 1
            continue
    
        if read == 1:
            link += char
        elif read == 3:
            name += char

    name = " ".join(name.split(" ")[:2]) # take 2 first words
    if name in names:
        continue
    names.append(name)

    female_pts = int(bash('<temp.txt grep -i -Fwo -- "she" | wc -l')) + int(bash('<temp.txt grep -i -Fwo -- "her" | wc -l'))
    male_pts = int(bash('<temp.txt grep -i -Fwo -- "he" | wc -l')) + int(bash('<temp.txt grep -i -Fwo -- "his" | wc -l'))

    if female_pts > male_pts:
        female_count += 1
        os.system("curl -s " + link + " > female" + str(female_count) + ".jpg")
        os.system("convert female" + str(female_count) + ".jpg -resize 45x45\> female" + str(female_count) + ".jpg")
    elif male_pts > female_pts:
        male_count += 1
        os.system("curl -s " + link + " > male" + str(male_count) + ".jpg")
        os.system("convert male" + str(male_count) + ".jpg -resize 45x45\> male" + str(male_count) + ".jpg")

os.system("rm temp.txt")  
