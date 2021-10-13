#encoding=utf-8
import MySQLdb
import xlwt
import xlrd

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

conn = MySQLdb.connect(host='localhost', user='root', passwd='root', port=3306)
cur = conn.cursor()
conn.select_db('Kikbug')
cur.execute("SET character_set_client = utf8;")
cur.execute("SET character_set_connection = utf8;")
cur.execute("SET character_set_results = utf8;")
cur.execute("SET character_set_server = utf8;")


# #update training_set中人工标注结果
# sql = "select bug.id,bug.bug_category3 from bug,report where bug.report_id = report.id and report.case_id = 10010000000019 order by bug.id"
# cur.execute(sql)
# res = cur.fetchall()
# conn.commit()
#
# for i in res:
#     id = i[0]
#     category = i[1]
#     sql = "update training_set set bug_category1 = %s where bug_id = %s"
#     cur.execute(sql,[category,id])
#     conn.commit()
# #


## 导出人工标注到Excel
# sql = "select bug.id, bug.report_id, bug.create_time_millis, bug.bug_category, bug.description, bug.img_url, bug.severity, bug.recurrent, bug.from_cloud, bug.isvalid, bug.bug_category3 from bug,report where bug.report_id = report.id and report.case_id = 10010000000025 order by bug.id"
# cur.execute(sql)
# result = cur.fetchall()
# conn.commit()
#
# wbk = xlwt.Workbook()
# sheet = wbk.add_sheet('Sheet1', cell_overwrite_ok=True)
#
# flag_i = 0
# flag_j = 0
# for i in result:
#     for j in i:
#         if flag_j == 0 or flag_j == 1:
#             sheet.write(flag_i, flag_j, str(j).replace('L', '').replace('(', '').replace(')', '').replace(',', ''))
#         elif flag_j == 2 or flag_j == 6 or flag_j == 7 or flag_j == 8 or flag_j == 9 or flag_j == 10:
#             sheet.write(flag_i, flag_j, str(j).decode('utf-8'))
#         elif flag_j == 3 or flag_j == 4 or flag_j == 11:
#             sheet.write(flag_i, flag_j, str(j).decode('utf-8'))
#         elif flag_j == 5:
#             tmp = str(j).decode('utf-8').split('/')
#             sheet.write(flag_i, flag_j, tmp[len(tmp)-1])
#
#         flag_j += 1
#     flag_j = 0
#     flag_i += 1
#     # for j in xrange(len(list(result[i]))):
#     #     sheet.write(i, j, result[i][j])
#     wbk.save("/Users/liyuying/Desktop/10010000000025" + '.xls')


# #读取excel中人工标注信息存入sql

# workbook = xlrd.open_workbook(r'/Users/liyuying/Desktop/app/HuaWei.xls')
# sheet = workbook.sheet_by_name('HuaWei')
# for i in range(1, sheet.nrows):
#     id = str(sheet.row_values(i)[0])
#     category = str(sheet.row_values(i)[10])
#     is_valid = str(sheet.row_values(i)[9])
#     sql = "update bug set bug_category3 = %s,isvalid=%s where id = %s"
#     cur.execute(sql, [category,is_valid, id])
#     conn.commit()


# #insert training_set

sql = "select bug.id, bug.bug_category3, report.case_id, bug.description,bug.img_url from bug,report where bug.report_id = report.id and report.case_id = 10010000000032 order by bug.id"
cur.execute(sql)
result = cur.fetchall()
conn.commit()

for one in result:
    bug_id = str(one[0]).replace('L', '').replace('(', '').replace(')', '').replace(',', '')
    bug_category1 = one[1]
    case_id = str(one[2]).replace('L', '').replace('(', '').replace(')', '').replace(',', '')
    description = one[3]
    img_url = one[4]
    sql = "insert into training_set (bug_id,bug_category1, case_id,description,img_url) values(%s,%s,%s,%s,%s)"
    cur.execute(sql, [bug_id, bug_category1, case_id, description, img_url])
    print bug_id
    print bug_category1
    print description
    conn.commit()


# #update training_set中isvalid
# sql = "select bug.id,bug.isvalid from bug,report where bug.report_id = report.id and report.case_id = 10010000000036 order by bug.id"
# cur.execute(sql)
# res = cur.fetchall()
# conn.commit()
#
# for i in res:
#     id = i[0]
#     is_valid = i[1]
#     sql = "update training_set set isvalid = %s where bug_id = %s"
#     cur.execute(sql, [is_valid, id])
#     conn.commit()

#

##

# sql = "select bug_id, bug_category1, case_id, description, img_url from training_set_HuJiang"
# cur.execute(sql)
# result = cur.fetchall()
# conn.commit()
#
# for one in result:
#     bug_id = str(one[0]).replace('L', '').replace('(', '').replace(')', '').replace(',', '')
#     bug_category1 = one[1]
#     case_id = str(one[2]).replace('L', '').replace('(', '').replace(')', '').replace(',', '')
#     description = one[3]
#     img_url = one[4]
#     sql = "insert into training_set (bug_id,bug_category1, case_id,description,img_url) values(%s,%s,%s,%s,%s)"
#     cur.execute(sql, [bug_id, bug_category1, case_id, description, img_url])
#     print bug_id
#     print bug_category1
#     print description
#     conn.commit()
#
# sql = "select bug_id,bug_category1 from training_set where case_id = %s and img_url != \"\" ORDER BY bug_id"
# cur.execute(sql,"10010000000037")
# res = cur.fetchall()
# conn.commit()
# for i in res:
#     print i[0]