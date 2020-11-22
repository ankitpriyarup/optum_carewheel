#!/bin/bash
cd commands
input="_i"
data=$(cat "$input")
curl -G -i "http://localhost:8081/v2/time/?&entity_name=time&timezone=UTC&source_language=en" \
    --data-urlencode "structured_value=" \
    --data-urlencode "fallback_value=" \
    --data-urlencode "bot_message=" \
    --data-urlencode "message=$data" \
    > _o
curl -G -i "http://localhost:8081/v2/time/?&entity_name=time&timezone=UTC&source_language=hi" \
    --data-urlencode "structured_value=" \
    --data-urlencode "fallback_value=" \
    --data-urlencode "bot_message=" \
    --data-urlencode "message=$data" \
    >> _o
curl -G -i "http://localhost:8081/v2/date/?&entity_name=date&timezone=UTC&source_language=en" \
    --data-urlencode "structured_value=" \
    --data-urlencode "fallback_value=" \
    --data-urlencode "bot_message=" \
    --data-urlencode "message=$data" \
    >> _o
curl -G -i "http://localhost:8081/v2/date/?&entity_name=date&timezone=UTC&source_language=hi" \
    --data-urlencode "structured_value=" \
    --data-urlencode "fallback_value=" \
    --data-urlencode "bot_message=" \
    --data-urlencode "message=$data" \
    >> _o
