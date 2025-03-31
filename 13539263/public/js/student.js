class StudentSystem {
    constructor() {
        this.submissions = [];
        // 在构造函数中获取当前登录学生的信息
        const studentName = localStorage.getItem('name');
        const studentId = localStorage.getItem('username');
        if (studentName && studentId) {
            document.getElementById('studentName').value = studentName;
            document.getElementById('studentId').value = studentId;
            this.fetchHistory(studentId);
        }
    }

    async submitAssignment(event) {
        event.preventDefault();
        
        const studentName = document.getElementById('studentName').value;
        const studentId = document.getElementById('studentId').value;
        const fileInput = document.getElementById('assignmentFile');

        if (!studentName || !studentId || !fileInput.files[0]) {
            alert('请填写完整信息并选择文件');
            return;
        }

        const file = fileInput.files[0];
        if (!this.validateFile(file)) {
            alert('请上传PDF或Word文件');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('studentName', studentName);
        formData.append('studentId', studentId);

        try {
            const submitButton = document.querySelector('.file-upload button');
            const originalText = submitButton.textContent;
            submitButton.textContent = '提交中...';
            submitButton.disabled = true;

            // 添加 token 到请求头
            const token = localStorage.getItem('token');
            if (!token) {
                alert('请先登录');
                window.location.href = '/login.html';
                return;
            }

            const response = await fetch('http://localhost:3000/api/submit', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`
                },
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.submissions.push(result.submission);
                this.updateHistory();
                
                fileInput.value = '';
                alert('作业提交成功！');
            } else {
                alert(result.message || '提交失败，请重试');
                if (result.message === '请先登录') {
                    window.location.href = '/login.html';
                }
            }

            submitButton.textContent = originalText;
            submitButton.disabled = false;
        } catch (error) {
            console.error('提交失败:', error);
            alert('提交失败，请重试');
            document.querySelector('.file-upload button').disabled = false;
        }
    }

    validateFile(file) {
        const validTypes = ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
        return validTypes.includes(file.type);
    }

    updateHistory() {
        const historyDiv = document.getElementById('submissionHistory');
        historyDiv.innerHTML = '';

        this.submissions.forEach(submission => {
            const div = document.createElement('div');
            div.className = 'submission-item';
            div.innerHTML = `
                <div class="submission-info">
                    <p>文件名：${submission.fileName}</p>
                    <p>提交时间：${submission.submitTime}</p>
                    <p>状态：${submission.status}</p>
                    ${submission.score ? `<p>分数：${submission.score}</p>` : ''}
                    ${submission.feedback ? `<p>评语：${submission.feedback}</p>` : ''}
                </div>
            `;
            historyDiv.appendChild(div);
        });
    }

    async fetchHistory(studentId) {
        try {
            const token = localStorage.getItem('token');
            if (!token) {
                console.error('未登录');
                return;
            }

            const response = await fetch(`http://localhost:3000/api/history/${studentId}`, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            const submissions = await response.json();
            this.submissions = submissions;
            this.updateHistory();
        } catch (error) {
            console.error('获取历史记录失败:', error);
        }
    }
}

const student = new StudentSystem();

// 修改全局提交函数
function submitAssignment(event) {
    student.submitAssignment(event);
}

// 页面加载完成后自动获取历史记录
window.onload = () => {
    const studentId = localStorage.getItem('username');
    if (studentId) {
        student.fetchHistory(studentId);
    }
}; 