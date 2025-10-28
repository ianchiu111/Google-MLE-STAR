
## entry point of the system
from MLE_Agent.graph import mle_star_process_tool

import logging
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s :: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("deep_research")

test_input = """
請幫我利用 python code 進行 Machine Learning，利用 data/store.csv 進行特徵工程並以 data/train.csv 中的 Sales 欄位為預測目標。
以下是資料集的描述，請幫我完成所有步驟。
並講所有相關的結果儲存在資料夾 data/information_from_agent/ 的資料夾中。

File 1:
data/train.csv - historical sales data

Data Schema:
1. Store - a unique Id for each store
2. Date - the date of the sales record
3. DayOfWeek - the day of the week
4. Customers - the number of customers on a given day
5. Sales - the turnover for any given day (this is what you are predicting)
6. Open - an indicator for whether the store was open: 0 = closed, 1 = open
7. StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
8. SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
9. Promo - indicates whether a store is running a promo on that day

---

File 2:
data/store.csv - supplemental information about the stores

Data Schema:
1. Store - a unique Id for each store
2. StoreType - differentiates between 4 different store models: a, b, c, d
3. Assortment - describes an assortment level: a = basic, b = extra, c = extended
4. CompetitionDistance - distance in meters to the nearest competitor store
5. CompetitionOpenSinceMonth - gives the approximate month of the time the nearest competitor was opened
6. CompetitionOpenSinceYear - gives the approximate year of the time the nearest competitor was opened
7. Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
8. Promo2SinceWeek - describes the calendar week when the store started participating in Promo2
9. Promo2SinceYear - describes the year when the store started participating in Promo2
10. PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
"""


if __name__ == "__main__":
    # first_test_input = "請幫我產生分析 dataframe 的 python code 範例，資料路徑為 'data/store.csv'。"

    response = mle_star_process_tool(test_input)
    logger.info("final response: %s", response)
