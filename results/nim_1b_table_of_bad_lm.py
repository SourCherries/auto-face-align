import os
import csv

expression = ["an", "ca", "di", "fe", "ha", "ne", "sa", "sp"]
mouth = ["o", "c"]
databases = []

my_project_path = os.path.dirname(os.path.abspath(__file__))

# Output to long format CSV with everything broken down.
#   MOUTH       [o c]
#   EXPRESSION  [an ... sp]
#   STRICT      [YES, NO]
#   NUMBER      The number of excluded

with open('table-DLIB-bad-landmarks-NIM.csv', 'w') as writer:
    # writer.write("Mouth,Expression,Strict,Number\n")
    writer.write("Strict,Mouth,Expression,Number\n")

    for mo in mouth:
        for ex in expression:
            dbase = "NIM-" + ex + "-" + mo
            my_faces_path = my_project_path + os.path.sep + dbase + os.path.sep

            with open(my_faces_path + "bad-landmarks-strict.csv") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    print("")
                num_exc_strict = len(row)
                writer.write('%s,%s,%s,%d\n' % ("yes", mo, ex, num_exc_strict))

            with open(my_faces_path + "bad-landmarks.csv") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    print("")
                num_exc_strict = len(row)
                writer.write('%s,%s,%s,%d\n' % ("no", mo, ex, num_exc_strict))






# END -------------------------------------------------------------------------
# -----------------------------------------------------------------------------
