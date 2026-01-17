async function test() {
    try {
        const response = await fetch("http://172.30.1.42:8000/v1/chat/completions", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                "messages": [
                    {
                        "role": "user",
                        "content": "한국어로 반갑게 인사해줘? /no_think"
                    }
                ],
                "stream": false,
                "chat_template_kwargs": {
                    "enable_thinking": false
                }
            })
        });

        const payload = await response.json();
        console.log(payload.choices[0].message);
        
    } catch (error) {
        console.error("에러가 발생했습니다:", error);
    }
}

test();