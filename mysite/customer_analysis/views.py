# 视图函数
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.shortcuts import render

import pandas as pd 
import numpy as np
import os

# sklearn相关
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# pyechart相关
from pyecharts.charts import *
from pyecharts import options as opts
from pyecharts.globals import ThemeType

import warnings
warnings.filterwarnings('ignore')

# ------------------------------------     全局变量     -------------------------------------------
TARGET_FILE = None # 上传上来的文件名

demo_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents'] # Demographic人口统计学的

serv_features = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup'
                , 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'] # 服务相关数据项

cat_accinfo_features = ['Contract', 'PaperlessBilling', 'PaymentMethod'] # 账户数据项 - 非数字

num_accinfo_features = ['tenure', 'MonthlyCharges', 'TotalCharges'] # 账户数据项 - 数字

# ------------------------------------------------------------------------------------------------

# 根目录直接重定向到home
def rd(request):
    return redirect('/home')

# 显示home.html
def home(request):
    return render(request, 'home.html')

# 上传成功后显示success文件
def success(request):
    return render(request, 'success.html')

# 存储上传的文件，文件名为TARGET_FILE
def uploading(request):
    rec_file = request.FILES.get('files[]')
    global TARGET_FILE
    TARGET_FILE = rec_file.name
    with open(f'upload/{rec_file.name}','wb') as f:
        f.write(rec_file.read())
    return JsonResponse({"status":0})

# csv转html表格，并在前端显示
def show_csv(request):
    if TARGET_FILE != None:
        data = pd.read_csv(f"upload/{TARGET_FILE}", encoding="gb18030")
        data_head = data.head(100)
        data_list = data_head.to_html(border=5, classes="table table-hover table-striped")
        return render(request, 'show_csv.html', {'data_list':data_list})
    else:
        return redirect('/home')

# 数据处理，缺失值处理，计算describe()和样本每一列有多少不同值，并输出显示
def data(request):
    if TARGET_FILE != None:
        df = pd.read_csv(f"upload/{TARGET_FILE}", encoding="gb18030")
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce') 
        df.loc[:,'TotalCharges'] = df.loc[:,'TotalCharges'].replace(np.nan,0)
        df['SeniorCitizen'].replace(0, 'No', inplace=True)
        df['SeniorCitizen'].replace(1, 'Yes', inplace=True)
        df.to_csv(path_or_buf="upload/save.csv")
        data = df.describe().T
        data_list_1 = data.to_html(border=5, classes="table table-hover table-striped text-center", justify="center")
        data_list_2 = []
        for col in df.select_dtypes('object').columns:
            ls = []
            ls.append(col)
            ls.append(df[col].nunique())
            data_list_2.append(ls)
        df_data_list_2 = pd.DataFrame(data_list_2)
        data_list_2 = df_data_list_2.to_html(border=5, classes="table table-hover table-striped text-center", justify="center")
        context = {
            'data_list_1':data_list_1,
            'data_list_2':data_list_2,
        }
        return render(request, 'data.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})

# 目标变量分析图
def chart(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df["Churn"].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar(init_opts=opts.InitOpts(width='400px', height='400px'))
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="客户流失数据表"))

        )
        pie = (
            Pie(init_opts=opts.InitOpts(width='400px', height='400px'))
            .add('', [list(z) for z in zip(x_data, y_data)],radius = ["40%", "70%"])
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="客户流失百分比",
                    pos_left="center",
                    pos_top="20",
                    title_textstyle_opts=opts.TextStyleOpts(color="#000"),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
            )   
        )
        context = {
            'bar':bar.render_embed(),
            'pie':pie.render_embed(),
        }
        return render(request, 'chart.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})

# -------------------------   以一下是各项的图表   -----------------------------------

def gender(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['gender'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="gender"))
        )

        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


def SeniorCitizen(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['SeniorCitizen'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="SeniorCitizen"))
        )
        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


def Partner(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['Partner'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="Partner"))
        )
        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


def Dependents(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['Dependents'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="Dependents"))
        )
        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


def PhoneService(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['PhoneService'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="PhoneService"))
        )
        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


def MultipleLines(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['MultipleLines'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="MultipleLines"))
        )
        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


def InternetService(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['InternetService'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="InternetService"))
        )
        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


def OnlineSecurity(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['OnlineSecurity'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="OnlineSecurity"))
        )
        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


def OnlineBackup(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['OnlineBackup'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="OnlineBackup"))
        )
        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


def DeviceProtection(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['DeviceProtection'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="DeviceProtection"))
        )
        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


def TechSupport(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['TechSupport'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="TechSupport"))
        )
        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


def StreamingTV(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['StreamingTV'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="StreamingTV"))
        )
        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


def StreamingMovies(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['StreamingMovies'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="StreamingMovies"))
        )
        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


def Contract(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['Contract'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="Contract"))
        )
        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


def PaperlessBilling(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['PaperlessBilling'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="PaperlessBilling"))
        )
        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


def PaymentMethod(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['PaymentMethod'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="PaymentMethod"))
        )
        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


def tenure(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['tenure'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="tenure"))
        )
        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


def MonthlyCharges(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['MonthlyCharges'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="MonthlyCharges"))
        )
        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


def TotalCharges(request):
    if os.path.exists('upload/save.csv'):
        df = pd.read_csv("upload/save.csv")
        col = df['TotalCharges'].value_counts()
        x_data = col.index.tolist()
        y_data = col.values.tolist()
        bar = (
            Bar({"theme": ThemeType.ROMA})
            .add_xaxis(x_data)
            .add_yaxis('Number', y_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="TotalCharges"))
        )
        context = {
            'bar':bar.render_embed(),
        }
        return render(request, 'other_charts.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})



# ----------------------------------------------------------------------

# 数据处理
def shuju_chuli():
    # 复制数据集并将特征与目标分开。
    df = pd.read_csv("upload/save.csv")
    X = df.copy().drop('Churn', axis=1)
    Y = df['Churn'].copy()

    # 删除customer ID
    X = X.drop(['customerID'], axis = 1)

    # 所有数据转为数字
    gender_map = {'Female': 0, 'Male': 1}
    yes_or_no_map = {'No': 0, 'Yes': 1} #seniorcitizen, partner, dependents, phoneservice, paperlessbilling
    multiplelines_map = {'No phone service': -1, 'No': 0, 'Yes': 1}
    internetservice_map = {'No': -1, 'DSL': 0, 'Fiber optic': 1}
    add_netservices_map = {'No internet service': -1, 'No': 0, 'Yes': 1} #onlinesecurity, onlinebackup, deviceprotection,techsupport,streaming services
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    paymentmethod_map = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}

    # 转int
    X['gender'] = X['gender'].map(gender_map).astype('int')
    X['Partner'] = X['Partner'].map(yes_or_no_map).astype('int')
    X['SeniorCitizen'] = X['SeniorCitizen'].map(yes_or_no_map).astype('int')
    X['Dependents'] = X['Dependents'].map(yes_or_no_map).astype('int')
    X['PhoneService'] = X['PhoneService'].map(yes_or_no_map).astype('int')
    X['MultipleLines'] = X['MultipleLines'].map(multiplelines_map).astype('int')
    X['InternetService'] = X['InternetService'].map(internetservice_map).astype('int')
    X['OnlineSecurity'] = X['OnlineSecurity'].map(add_netservices_map).astype('int')
    X['OnlineBackup'] = X['OnlineBackup'].map(add_netservices_map).astype('int')
    X['DeviceProtection'] = X['DeviceProtection'].map(add_netservices_map).astype('int')
    X['TechSupport'] = X['TechSupport'].map(add_netservices_map).astype('int')
    X['StreamingTV'] = X['StreamingTV'].map(add_netservices_map).astype('int')
    X['StreamingMovies'] = X['StreamingMovies'].map(add_netservices_map).astype('int')
    X['Contract'] = X['Contract'].map(contract_map).astype('int')
    X['PaperlessBilling'] = X['PaperlessBilling'].map(yes_or_no_map).astype('int')
    X['PaymentMethod'] = X['PaymentMethod'].map(paymentmethod_map).astype('int')

    # 现在我们将数据分成训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42
                                                    , stratify = Y)
    list = [X_train, Y_train, X_test, Y_test, X, Y]
    return list

# 显示训练集
def X_train(request):
    if os.path.exists('upload/save.csv'):
        list = shuju_chuli()
        data = list[0].to_html(index=False, border=5, classes="table table-hover table-striped text-center", justify="center")
        context = {
            'data':data,
            'title':"训练集"
        }
        return render(request, 'train_test.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})

# 训练集目标样本值
def Y_train(request):
    if os.path.exists('upload/save.csv'):
        list = shuju_chuli()[1]
        list = {'index':list.index,'churn':list.values}
        df_list = pd.DataFrame(list) 
        data = df_list.to_html(border=5, classes="table table-hover table-striped text-center", justify="center")
        context = {
            'data':data,
            'title':"训练集样本结果"
        }
        return render(request, 'train_test.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


# 测试集
def X_test(request):
    if os.path.exists('upload/save.csv'):
        list = shuju_chuli()
        data = list[2].to_html(index=False, border=5, classes="table table-hover table-striped text-center", justify="center")
        context = {
            'data':data,
            'title':"测试集"
        }
        return render(request, 'train_test.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})

# 测试集目标样本值
def Y_test(request):
    if os.path.exists('upload/save.csv'):
        list = shuju_chuli()[3]
        list = {'index':list.index,'churn':list.values}
        df_list = pd.DataFrame(list) 
        data = df_list.to_html(border=5, classes="table table-hover table-striped text-center", justify="center")
        context = {
            'data':data,
            'title':"测试集样本结果"
        }
        return render(request, 'train_test.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})

# 逻辑回归
def Logistic_Regression(request):
    if os.path.exists('upload/save.csv'):
        list = shuju_chuli()
        X_train = list[0]
        Y_train = list[1]
        X_test = list[2]
        Y_test = list[3]
        X = list[4]
        num_features = num_accinfo_features

        cat_3p_features = []
        for col in X.columns:
            if (X[col].nunique() > 2) & (X[col].nunique() < 5):  #less than 5 to exclude the numerical features
                cat_3p_features.append(col)
        cat_transformer = OneHotEncoder(handle_unknown='ignore')
        num_transformer = StandardScaler()
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_features),
                ('cat', cat_transformer, cat_3p_features)      
            ], remainder='passthrough')
        lr_pipe = Pipeline([('Transformers', preprocessor)
                            ,('LR',  LogisticRegression(random_state = 42, max_iter = 1000))])
        lr_param = {'LR__C': 3.0}
        lr_pipe.set_params(**lr_param) 
        lr_pipe.fit(X_train, Y_train)
        pred_lr = lr_pipe.predict(X_test)
        value = []
        value.append(pred_lr)
        value.append(Y_test)
        df_value = pd.DataFrame(value).T
        df_value.columns = ["测试集结果", "测试集样本"]
        data = df_value.to_html(border=5, classes="table table-hover table-striped text-center", justify="center")
        percent = str.format("测试精度：{0:.6f}%", metrics.accuracy_score(Y_test, pred_lr)*100)
        print(percent)
        context = {
            'data':data,
            'title':"训练数据比对结果",
            'percent':percent,
        }
        return render(request, 'show_value.html', context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})


# 等待页面
def waiting(request):
    if os.path.exists('upload/save.csv'):
        return render(request, "waiting.html")
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})

# 选择输入索引页面
def select_show(request):
    return render(request, 'select_show.html')

# 最后显示结果页面
def final(request):
    if os.path.exists('upload/save.csv'):
        index = request.POST.get('number')
        list = shuju_chuli()
        X_train = list[0]
        Y_train = list[1]
        X = list[4]
        Y = list[5]
        num_features = num_accinfo_features
        cat_3p_features = []
        for col in X.columns:
            if (X[col].nunique() > 2) & (X[col].nunique() < 5):  #less than 5 to exclude the numerical features
                cat_3p_features.append(col)
        cat_transformer = OneHotEncoder(handle_unknown='ignore')
        num_transformer = StandardScaler()
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_features),
                ('cat', cat_transformer, cat_3p_features)      
            ], remainder='passthrough')
        lr_pipe = Pipeline([('Transformers', preprocessor)
                            ,('LR',  LogisticRegression(random_state = 42, max_iter = 1000))])
        lr_param = {'LR__C': 3.0}
        lr_pipe.set_params(**lr_param) 
        lr_pipe.fit(X_train, Y_train)
        n = int(index)
        table = X[n:n+1]
        pred_lr = lr_pipe.predict(table)
        # table = table.T
        table_input = table.to_html( border=5, classes="table table-hover table-striped text-center", justify="center")
        output = pred_lr[0]
        right_val = Y[n]
        context = {
            "output":output,
            "right_val":right_val,
            "table_input":table_input,
        }
        
        return render(request, "final.html", context)
    else:
        return render(request, 'home_tmp.html', {'alert':"您还没上传数据！点击确定跳转"})