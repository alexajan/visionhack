from utils import read_files_and_tags

def merge(filenames):
    # bridge-0, city_entry-1, city_exit-2, bump-3, wipers-4, zebra-5
    a = []
    for _file in filenames:
        files_and_tags = read_files_and_tags(_file)
        a.append(list(map(lambda x: x[1], files_and_tags)))
    files = list(map(lambda x: x[0] + ' ', files_and_tags))
    
    for i in range(len(a)):
        for j in range(len(a[0])):
            if i == 2:
                files[j] += '0' if int(a[0][j][0]) else a[i][j][i]
            else:
                files[j] += a[i][j][i]
    with open('submission.txt', 'w') as s:
        s.write('\n'.join(files))


# merge(['/Users/emmanuelazuh/Downloads/trainset/train.txt','/Users/emmanuelazuh/Downloads/trainset/train.txt'])
