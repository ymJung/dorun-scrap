from datetime import datetime, timedelta
import pymysql
import win32com.client as com
import configparser


class Store:
    def __init__(self):
        cf = configparser.ConfigParser()
        cf.read('config.cfg')
        self.connection = pymysql.connect(host=cf.get('db', 'DB_IP'),
                                          user=cf.get('db', 'DB_USER'),
                                          password=cf.get('db', 'DB_PWD'),
                                          db=cf.get('db', 'DB_SCH'),
                                          charset='utf8mb4',
                                          cursorclass=pymysql.cursors.DictCursor)
        self.DEFAULT_FIRST_DATE = 20040101
        self.KOSPI_200 = 180
        self.stock_chart = com.Dispatch("CpSysDib.StockChart")
        self.stock_chart.SetInputValue(1, ord('1'))
        self.stock_chart.SetInputValue(5,
                                       (0, 2, 3, 4, 5, 8, 16, 21))
        self.stock_chart.SetInputValue(6, ord('D'))
        self.stock_chart.SetInputValue(9, ord('1'))
        self.code_mgr = com.Dispatch("CpUtil.CpCodeMgr")

    def __del__(self):
        self.connection.close()

    def commit(self):
        self.connection.commit()

    def save_stocks(self, code, ds_stock_chart):
        cursor = self.connection.cursor()
        for i in range(ds_stock_chart.GetHeaderValue(3)):
            date = ds_stock_chart.GetDataValue(0, i)
            cursor.execute(
                "INSERT INTO "
                "data.daily_stock(code, date, open, high, low, close, volume, hold_foreign, st_purchase_inst) "
                "VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    code,
                    datetime(date // 10000, date // 100 % 100, date % 100),
                    ds_stock_chart.GetDataValue(1, i),
                    ds_stock_chart.GetDataValue(2, i),
                    ds_stock_chart.GetDataValue(3, i),
                    ds_stock_chart.GetDataValue(4, i),
                    ds_stock_chart.GetDataValue(5, i),
                    float(ds_stock_chart.GetDataValue(6, i)),
                    float(ds_stock_chart.GetDataValue(7, i))
                )
            )
            print('saved stocks code ', code, ' date', date)
        self.commit()

    def get_possible_store_date(self, code):
        cursor = self.connection.cursor()
        cursor.execute("SELECT date FROM data.daily_stock WHERE code = %s ORDER BY date DESC LIMIT 1", code)
        last_date = cursor.fetchone()
        if last_date is None:
            return self.DEFAULT_FIRST_DATE
        after_day = datetime.strptime(last_date, "%Y-%m-%d %H:%M:%S") + timedelta(days=1)
        return after_day.year * 10000 + after_day.month * 100 + after_day.day

    def is_invalid_status(self):
        if self.stock_chart.BlockRequest() != 0 or self.stock_chart.GetDibStatus() != 0:
            print('invalid status.')
            return True
        return False

    def run(self):
        for code in self.code_mgr.GetGroupCodeList(self.KOSPI_200):
            possible_store_date = self.get_possible_store_date(code)
            self.stock_chart.SetInputValue(0, code)
            self.stock_chart.SetInputValue(3, possible_store_date)
            if self.is_invalid_status():
                continue
            if self.stock_chart.GetHeaderValue(5) < possible_store_date:
                continue
            self.save_stocks(code, self.stock_chart)

            while self.stock_chart.Continue:
                if self.is_invalid_status():
                    continue
                self.save_stocks(code, self.stock_chart)


Store().run()