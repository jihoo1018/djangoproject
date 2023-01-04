from admin.dlearn.aitrader.models import Kospi

menu = ["Exit",
        "Dnn Model",
        "Lstm Model",
        "Dnn Ensemble",
        "Lstm Ensemble"]
if __name__ == '__main__':
    model = Kospi()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                if menu == "1": model.create_dnn()
                elif menu == "2": model.create_lstm()
                elif menu == "3": model.dnnensemble()
                elif menu == "4": model.lstmensemble()
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message')
                else:
                    print("Didn't catch error message")