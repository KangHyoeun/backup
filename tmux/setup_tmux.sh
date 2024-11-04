#!/bin/bash
# setup_tmux.sh

# 세션 이름 정의
SESSION_NAME="mysession"

# tmux 세션이 이미 존재하는지 확인
tmux has-session -t $SESSION_NAME 2>/dev/null

# 세션이 없으면 새로 생성
if [ $? != 0 ]; then
  # 새로운 tmux 세션 생성 (백그라운드로 실행)
  tmux new-session -d -s $SESSION_NAME

  # 첫 번째 창에서 .profile 소싱하고 도커 컨테이너에 진입
  tmux send-keys -t $SESSION_NAME "source ~/.profile" C-m
  tmux send-keys -t $SESSION_NAME "vrx" C-m

  # 잠시 대기하여 컨테이너가 실행될 시간을 확보
  sleep 2

fi

# tmux 세션을 attach 하여 사용자에게 보여줍니다.
tmux attach -t $SESSION_NAME
