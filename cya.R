library('tidyverse')        #자료 불러오기
library('tidymodels')       #머신러닝 모델

cya <- read_csv('cya.csv') # 2009년부터 2019년까지 규정 이닝을 채운 투수들의 정보

cya

cya %>% 
  filter(season==2019) -> cya_2019   # 19년도 사이 수상 결과가 없기 때문에 제외


cya %>% 
  filter(season!=2019) -> cya_pre    # 2009~2018년도 데이터를 별도로 저장 


cya_pre %>%
  initial_split(prop=0.7) -> cya_split  # 70퍼를 학습용으로 
cya_split       # 학습용 데이터 525, 시험용 데이터 226, 전체 데이터 751


cya_split %>%
  training()            #학습용 데이터 확인

cya_split %>%
  testing()             #시험용 데이터 확인


cya_split %>% training() %>%
  recipe(vote~w+l+war+era+fip+ip+k+bb+hr+k9+bb9+kbb+babip)    # 결과 vote, 나머지 13개 변수를 예측에 사용


cya_split %>% training() %>%
  recipe(vote~w+l+war+era+fip+ip+k+bb+hr+k9+bb9+kbb+babip) %>%
  step_corr(all_predictors()) %>%                   # step_corr()함수로 상관관계가 지나치게 큰 변수를 제거
  step_center(all_predictors(), -all_outcomes()) %>%# step_center()와 step_scale() 함수로 평군을 0으로 하는 척도를 만듬
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep() -> cya_recipe


cya_recipe        #확인
summary(cya_recipe)

cya_recipe %>%
  bake(cya_split %>% testing()) -> cya_testing       #bake()함수를 통해 시험용 데이터를 처리함.

cya_testing


cya_recipe %>%
  juice() -> cya_training                           #juice()함수로는 학습용 데이터를 처리함.

cya_training



rand_forest(trees=100, mode='regression') %>%      #trees 는 트리의수를 설정, mode는 모형 설정
  set_engine('randomForest') %>%                   #set_engine() 어떤 패키지를 사용할지 설정하는 것.
  fit(vote~w+l+war+era+ip+bb+hr+k9+kbb+babip, data=cya_training) -> cya_rf
  #fit()함수에 학습 데이터와 공식을 넣어서 추측을 함.
cya_rf



cya_rf %>%
  predict(cya_testing)%>%                 #predict() 함수로 예측값을 구함.
  bind_cols(cya_testing)%>%               #결과만 놔오면 헷갈리니 bind_cols()를 써서 합침.
  metrics(truth=vote, estimate=.pred)     #yardstick 패키지 안에 있는 metrics()함수를  써서 성능을 확인함.


# 2018년 이전 전체 사이 영 상 수상자 예측하기

#새로운 레시피를 만든다.
cya_pre %>%
  recipe(vote~w+l+war+era+fip+ip+k+bb+hr+k9+bb9+kbb+babip) %>%
  step_corr(all_predictors()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep() -> cya_recipe2


cya_recipe2


# 시험용 데이터를 bake() 한다.
cya_recipe2 %>%
  bake(cya_pre) -> cya_testing_pre


# 학습용 데이터를 juice() 한다.
cya_recipe2 %>%
  juice() -> cya_training_pre


# 새로운 데이터로 랜덤 포레스트모형을 만든다.
rand_forest(trees=100, mode='regression') %>%
  set_engine('randomForest', localImp=TRUE) %>%
  fit(vote~w+l+war+era+ip+bb+hr+k9+kbb+babip, data=cya_training_pre) -> cya_rf2


#원래 데이터와 열을 합쳐 보면 결과가 나온다.
cya_rf2 %>%
  predict(cya_testing_pre) %>%
  bind_cols(cya_pre)


#최고점을 예상한 선수가 실제로 상을 탔는지를 알기 위해 
#먼저 시즌과 리그에 따라 모형 예상값 순위를 매김
#예상값이 1등이거나 아니면 사이영상을 탄, 
#즉 cy변수가 O가 들어있는 행만 골라낸다.

cya_rf2 %>%
  predict(cya_testing_pre) %>%
  bind_cols(cya_pre) %>%
  group_by(season, lg) %>%         #group_by(): 출력결과를 데이터프레임의 업그레이드 버전인 tibble 형태로 만들어줌
  mutate(rank=rank(-.pred)) %>%    #mutate(): 열을 새로 만듬.
  select(season, lg, team, name, cy, rank) %>%   #select(): 해당 변수들을 추출
  filter(rank==1 | cy=='O') %>%     # rank가 1 또는 cy가 'O'인거를 뽑음.
  arrange(season, lg, cy)          # arrange(): 정렬을 도우는 함수.



#2019예측 해보기
#같은 모형을 2019년도 자료에 적용을 함.

cya_recipe2 %>%
  bake(cya_2019) -> cya_testing_2019


#어떤 선수가 상위권에 포진했는지 확인.
cya_rf2 %>%
  predict(cya_testing_2019) %>%
  bind_cols(cya_2019) %>%
  arrange(-.pred) %>%
  select(name, lg, team, .pred)

#사이영상 수상에서의 중요한 기록 찾기
install.packages('randomForestExplainer')
library('randomForestExplainer')



measure_importance(cya_rf2$fit)


measure_importance(cya_rf2$fit) %>%
  as_tibble() %>%
  mutate(imp=node_purity_increase*100/max(node_purity_increase)) %>%
  arrange(-imp) %>%
  select(variable, imp)
