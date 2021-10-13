# import xlwt
# import xlrd

# workbook = xlrd.open_workbook('/Users/liyuying/Desktop/result1.xls')
# sheet = workbook.sheet_by_index(2)

# data = [sheet.cell_value(1, col) for col in range(sheet.ncols)]

# workbook = xlwt.Workbook()
# sheet = workbook.add_sheet('test')

# for index, value in enumerate(data):
#     sheet.write(0, index, value)

# workbook.save('/Users/liyuying/Desktop/result2.xls')

import xlwt
import xlrd
from xlutils.copy import copy
A = 5
alg = ['KNN', 'NB', 'RF', 'DT', 'SVM']
# ACC
workbook = xlrd.open_workbook('/Users/liyuying/Desktop/result1.xls')
# wb = copy(workbook)
# rb = xlrd.open_workbook('/Users/liyuying/Desktop/result2.xls')
workbook_w = xlwt.Workbook()
for method in xrange(0,A):
	sheet = workbook_w.add_sheet(alg[method])
	sheet_txt = workbook.sheet_by_index(method + 1)
	sheet_all = workbook.sheet_by_index(method + A + 1)
	data = [0,0,0,0,0,0,0,0,0,0,0,0]
	for app in xrange(0,6):
		data[2 * app] = [sheet_txt.cell_value(row, 1 + 2 * app) for row in range(sheet_txt.nrows)]
		data[2 * app + 1] = [sheet_all.cell_value(row, 1 + 2 * app) for row in range(sheet_all.nrows)]
    # sheet = workbook_w.add_sheet(str(alg[method]))
	for col in xrange(0,12):
		for row in xrange(0,31):
			value = data[col][row]
			# for index, value in enumerate(data[col]):
			sheet.write(row, col, value)
			# print "index:" + str(index)
			# print "col:" + str(col)
			# print "value:" + str(value)
	    	
	workbook_w.save('/Users/liyuying/Desktop/result2.xls')