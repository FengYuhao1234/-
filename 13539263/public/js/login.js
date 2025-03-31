let currentType = 'student';

function switchType(type) {
    currentType = type;
    const studentBtn = document.getElementById('studentBtn');
    const teacherBtn = document.getElementById('teacherBtn');
    
    if (type === 'student') {
        studentBtn.classList.add('active');
        teacherBtn.classList.remove('active');
    } else {
        teacherBtn.classList.add('active');
        studentBtn.classList.remove('active');
    }

    // 切换时添加动画效果
    const form = document.querySelector('.login-form');
    form.style.opacity = '0';
    setTimeout(() => {
        form.style.opacity = '1';
    }, 200);
}

async function handleLogin(event) {
    event.preventDefault();
    
    const submitButton = event.target.querySelector('button');
    const originalText = submitButton.innerHTML;
    submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 登录中...';
    submitButton.disabled = true;
    
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    try {
        const response = await fetch('http://localhost:3000/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username,
                password,
                type: currentType
            })
        });

        const result = await response.json();
        
        if (result.success) {
            submitButton.innerHTML = '<i class="fas fa-check"></i> 登录成功';
            
            // 保存登录信息
            localStorage.setItem('token', result.token);
            localStorage.setItem('userType', currentType);
            localStorage.setItem('username', username);
            localStorage.setItem('name', result.user.name);

            // 延迟跳转，显示成功状态
            setTimeout(() => {
                window.location.href = currentType === 'student' ? '/student.html' : '/teacher.html';
            }, 1000);
        } else {
            throw new Error(result.message || '登录失败');
        }
    } catch (error) {
        console.error('登录失败:', error);
        submitButton.innerHTML = '<i class="fas fa-exclamation-circle"></i> 登录失败';
        setTimeout(() => {
            submitButton.innerHTML = originalText;
            submitButton.disabled = false;
        }, 2000);
        alert(error.message || '登录失败，请重试');
    }
}

// 页面加载时设置默认选中状态
window.onload = () => {
    switchType('student');
    
    // 添加输入框动画效果
    const inputs = document.querySelectorAll('input');
    inputs.forEach(input => {
        input.addEventListener('focus', () => {
            input.parentElement.style.transform = 'translateY(-2px)';
        });
        input.addEventListener('blur', () => {
            input.parentElement.style.transform = 'translateY(0)';
        });
    });
}; 