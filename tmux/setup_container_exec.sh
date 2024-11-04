#!/bin/bash
# setup_container_exec.sh

# 세션 이름 정의
SESSION_NAME="newsession"

# tmux 세션이 이미 존재하는지 확인
tmux has-session -t $SESSION_NAME 2>/dev/null

# 세션이 없으면 새로 생성
if [ $? != 0 ]; then
    # 새로운 tmux 세션 생성 (백그라운드로 실행)
    tmux new-session -d -s $SESSION_NAME

    # 도커 컨테이너 ID 가져오기 (컨테이너 이름의 일부 또는 조건에 맞춰 선택)
    CONTAINER_ID=$(docker ps -q -f "name=dockwater_humble_runtime")

    if [ -z "$CONTAINER_ID" ]; then
    echo "No running container found with the specified name."
    exit 1
    fi

    # 창에 컨테이너로 진입하는 명령어 보내기
    # 수동으로 창을 이동해서 실행할 수 있도록 설정
    tmux send-keys -t $SESSION_NAME "docker exec -it $CONTAINER_ID /bin/bash" C-m

    sleep 2

fi

# tmux 세션을 attach 하여 사용자에게 보여줍니다.
tmux attach -t $SESSION_NAME