# Problem 1 report

## Problem 1.1

**Used LLM** :  Gemini, Chat GPT

**Code** :
_Write down your code from First LLM_  

### Gemini

    def learn_mul(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        """
        multi-class logistic model을 학습하고 반환합니다.
        Args:
            X: 학습 데이터의 특징 행렬
            y: 학습 데이터의 레이블 벡터
        Returns:
            학습된 LogisticRegression 모델
        """
        lr = LogisticRegression(random_state=42, solver='lbfgs', multi_class='multinomial', max_iter=1000)
        lr.fit(X, y)
        return lr


    def inference_mul(x: np.ndarray, lr_model: LogisticRegression) -> np.ndarray:
        """
        학습된 모델을 사용하여 예측값을 반환합니다.
        Args:
            x: 예측할 데이터의 특징 행렬
            lr_model: 학습된 LogisticRegression 모델
        Returns:
            예측된 레이블 벡터
        """
        y_pred = lr_model.predict(x)
        return y_pred

### Chat GPT

    def learn_mul(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        from sklearn.preprocessing import StandardScaler
        # 데이터 정규화 (표준화)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
        # 다중 클래스 분류를 위한 로지스틱 회귀 모델 생성
        lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
    
        # 모델 학습
        lr.fit(X_scaled, y)
    
        return lr

    
    def inference_mul(x: np.ndarray, lr_model: LogisticRegression) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler
        # 입력 데이터 표준화 (학습할 때 사용한 scaler 적용)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
    
        # 모델을 사용하여 예측 수행
        y_pred = lr_model.predict(x_scaled)
    
        return y_pred

**Code Description** :  
_Write down the description of the code from First LLM_




## Problem 1.2

**Used LLM** :  Gemini, Chat GPT

**Code** :  
_Write down your code from Second LLM_  

### Gemini

    def learn_mul2bin(X: np.ndarray, y: np.ndarray, num_classes: int) -> list[LogisticRegression]:
        """
        각 클래스에 대한 이진 분류 로지스틱 회귀 모델을 학습하고 반환합니다.
    
        Args:
            X: 학습 데이터의 특징 행렬
            y: 학습 데이터의 레이블 벡터
            num_classes: 클래스 수
    
        Returns:
            학습된 LogisticRegression 모델 리스트
        """
        lrs = []
        ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])
        for i in range(num_classes):
            print('training %s classifier'%(ordinal(i+1)))
            y_binary = (y.astype(int) == i).astype(int)
            lr = LogisticRegression(random_state=42, max_iter=10)
            lr.fit(X, y_binary)
            lrs.append(lr)
        return lrs


    def inference_mul2bin(X: np.ndarray, lrs: list[LogisticRegression]) -> np.ndarray:
        """
        학습된 이진 분류 모델들을 사용하여 예측값을 반환합니다.
    
        Args:
            X: 예측할 데이터의 특징 행렬
            lrs: 학습된 LogisticRegression 모델 리스트
    
        Returns:
            예측된 레이블 벡터
        """
        predictions = []
        for lr in lrs:
            predictions.append(lr.predict_proba(X)[:, 1])
        y_pred = np.argmax(np.array(predictions).T, axis=1)
        return y_pred

**Code Description** :  
_Write down the description of the code from Second LLM with a focus on differences from first LLM_

