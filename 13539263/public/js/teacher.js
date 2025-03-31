class TeacherSystem {
    constructor() {
        this.submissions = [];
        this.gradedWork = [];
        this.fetchGradedSubmissions(); // 新增：获取已批改作业
        this.fetchStudents(); // 获取学生列表
        this.autoGradingHistory = [];
        this.loadAutoGradingHistory();
    }

    // 获取待批改作业列表
    async fetchSubmissions() {
        try {
            const token = localStorage.getItem('token');
            if (!token) {
                console.error('未登录');
                return;
            }

            const response = await fetch('http://localhost:3000/api/submissions', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            this.submissions = await response.json();
            this.updateSubmissionsList();
        } catch (error) {
            console.error('获取作业列表失败:', error);
        }
    }

    // 新增：获取已批改作业列表
    async fetchGradedSubmissions() {
        try {
            const token = localStorage.getItem('token');
            if (!token) {
                console.error('未登录');
                return;
            }

            const response = await fetch('http://localhost:3000/api/graded-submissions', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            this.gradedWork = await response.json();
            this.updateGradedList();
        } catch (error) {
            console.error('获取已批改作业失败:', error);
        }
    }

    // 获取学生列表
    async fetchStudents() {
        try {
            const response = await fetch('http://localhost:3000/api/students', {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });
            const students = await response.json();
            this.updateStudentList(students);
        } catch (error) {
            console.error('获取学生列表失败:', error);
        }
    }

    // 添加学生
    async addStudent() {
        const username = document.getElementById('newStudentUsername').value;
        const password = document.getElementById('newStudentPassword').value;
        const name = document.getElementById('newStudentName').value;

        if (!username || !password || !name) {
            alert('请填写完整信息');
            return;
        }

        try {
            const response = await fetch('http://localhost:3000/api/students', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify({ username, password, name })
            });

            const result = await response.json();
            if (result.success) {
                alert('添加成功');
                this.fetchStudents();
                // 清空输入框
                document.getElementById('newStudentUsername').value = '';
                document.getElementById('newStudentPassword').value = '';
                document.getElementById('newStudentName').value = '';
            } else {
                alert(result.message || '添加失败');
            }
        } catch (error) {
            console.error('添加学生失败:', error);
            alert('添加失败，请重试');
        }
    }

    // 更新学生列表显示
    updateStudentList(students) {
        const studentList = document.getElementById('studentList');
        studentList.innerHTML = '';

        students.forEach(student => {
            const div = document.createElement('div');
            div.className = 'student-item';
            div.innerHTML = `
                <div class="student-info">
                    <p>用户名：${student.username}</p>
                    <p>姓名：${student.name}</p>
                </div>
                <div class="student-actions">
                    <button onclick="teacher.deleteStudent('${student.username}')">删除</button>
                </div>
            `;
            studentList.appendChild(div);
        });
    }

    // 删除学生
    async deleteStudent(username) {
        if (!confirm(`确定要删除学生 ${username} 吗？`)) {
            return;
        }

        try {
            const response = await fetch(`http://localhost:3000/api/students/${username}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });

            const result = await response.json();
            if (result.success) {
                alert('删除成功');
                this.fetchStudents();
            } else {
                alert(result.message || '删除失败');
            }
        } catch (error) {
            console.error('删除学生失败:', error);
            alert('删除失败，请重试');
        }
    }

    // 其他方法保持不变...

    // 修改提交评分方法
    async submitGrade() {
        const submissionId = document.getElementById('currentSubmission').dataset.submissionId;
        const score = document.getElementById('score').value;
        const feedback = document.getElementById('feedback').value;

        if (!submissionId) {
            alert('请先选择要评分的作业');
            return;
        }

        if (!score) {
            alert('请输入分数');
            return;
        }

        // 禁用提交按钮，防止重复提交
        const submitButton = document.querySelector('#manualGrading button');
        submitButton.disabled = true;
        submitButton.textContent = '提交中...';

        try {
            const token = localStorage.getItem('token');
            if (!token) {
                alert('请先登录');
                window.location.href = '/login.html';
                return;
            }

            const response = await fetch('http://localhost:3000/api/grade', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    submissionId: parseInt(submissionId),
                    score,
                    feedback
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                // 刷新待批改和已批改列表
                await this.fetchSubmissions();
                await this.fetchGradedSubmissions();
                
                // 清空评分表单
                document.getElementById('currentSubmission').innerHTML = '';
                document.getElementById('score').value = '';
                document.getElementById('feedback').value = '';
                
                // 返回到待批改列表
                switchSection('pending');
                
                // 显示成功提示
                alert('评分已提交');
            } else {
                throw new Error(result.message || '评分提交失败');
            }

        } catch (error) {
            console.error('提交评分失败:', error);
            alert('评分提交成功');
        } finally {
            // 恢复提交按钮状态
            submitButton.disabled = false;
            submitButton.textContent = '提交评分';
        }
    }

    // 计算文本相似度（使用余弦相似度）
    calculateSimilarity(text1, text2) {
        // 将文本转换为词频向量
        const getWordFrequency = (text) => {
            const words = text.toLowerCase().match(/\b\w+\b/g) || [];
            const frequency = {};
            words.forEach(word => {
                frequency[word] = (frequency[word] || 0) + 1;
            });
            return frequency;
        };

        const freq1 = getWordFrequency(text1);
        const freq2 = getWordFrequency(text2);
        const allWords = new Set([...Object.keys(freq1), ...Object.keys(freq2)]);

        // 计算向量的点积和模长
        let dotProduct = 0;
        let magnitude1 = 0;
        let magnitude2 = 0;

        allWords.forEach(word => {
            const f1 = freq1[word] || 0;
            const f2 = freq2[word] || 0;
            dotProduct += f1 * f2;
            magnitude1 += f1 * f1;
            magnitude2 += f2 * f2;
        });

        magnitude1 = Math.sqrt(magnitude1);
        magnitude2 = Math.sqrt(magnitude2);

        // 计算余弦相似度
        return magnitude1 && magnitude2 ? 
            (dotProduct / (magnitude1 * magnitude2)) * 100 : 0;
    }

    // 检查关键词匹配
    checkKeywords(text, keywords) {
        const matches = [];
        const lowercaseText = text.toLowerCase();
        
        keywords.forEach(keyword => {
            const keywordLower = keyword.toLowerCase().trim();
            if (lowercaseText.includes(keywordLower)) {
                matches.push(keyword.trim());
            }
        });
        
        return matches;
    }

    // 自动评分
    async autoGrade(submissionId) {
        const standardAnswer = document.getElementById('standardAnswer').value;
        const keywords = document.getElementById('keywords').value.split(',');
        const threshold = parseInt(document.getElementById('similarityThreshold').value);
        
        const submission = this.submissions.find(s => s.id === parseInt(submissionId));
        if (!submission) return;

        try {
            const token = localStorage.getItem('token');
            if (!token) {
                alert('请先登录');
                return;
            }

            // 先检查 Python 务是否在运行
            try {
                const healthCheck = await fetch('http://localhost:5000/health');
                if (!healthCheck.ok) {
                    throw new Error('自动评分服务未启动');
                }
            } catch (error) {
                alert('自动评分服务未启动，请确保 Python 服务正在运行');
                return;
            }

            // 调用 Python 服务进行自动评分
            const response = await fetch('http://localhost:5000/api/auto-grade', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    studentAnswer: submission.text || '',
                    standardAnswer,
                    keywords,
                    threshold
                })
            });

            if (!response.ok) {
                throw new Error('自动评分请求失败');
            }

            const result = await response.json();
            
            if (result.success) {
                // 显示评分结果
                document.getElementById('score').value = result.score;
                document.getElementById('feedback').value = result.feedback;
                
                // 切换到手动评分视图以便教师确认或修改
                switchGradingType('manual');
            } else {
                alert(result.message || '自动评分失败');
            }
        } catch (error) {
            console.error('自动评分失败:', error);
            alert(`自动评分失败: ${error.message}`);
        }
    }

    updateSubmissionsList() {
        const submissionsList = document.getElementById('submissionsList');
        submissionsList.innerHTML = '';

        this.submissions.forEach(submission => {
            const div = document.createElement('div');
            div.className = 'submission-item';
            div.innerHTML = `
                <div class="submission-info">
                    <h3>学生：${submission.studentName}</h3>
                    <p>学号：${submission.studentId}</p>
                    <p>提交时间：${submission.submitTime}</p>
                    <p>文件名：${submission.fileName}</p>
                </div>
                <div class="submission-actions">
                    <button onclick="teacher.viewSubmission(${submission.id})">查看详情</button>
                </div>
            `;
            submissionsList.appendChild(div);
        });

        // 更新工作台统计数据
        document.getElementById('pendingCount').textContent = this.submissions.length;
        document.getElementById('gradedCount').textContent = this.gradedWork.length;
    }

    async viewSubmission(submissionId) {
        const submission = this.submissions.find(s => s.id === submissionId);
        if (!submission) return;

        const currentSubmission = document.getElementById('currentSubmission');
        currentSubmission.dataset.submissionId = submissionId;

        // 创建预览区域
        let previewHtml = `
            <div class="submission-detail">
                <h3>作业详情</h3>
                <p><strong>学生：</strong>${submission.studentName}</p>
                <p><strong>学号：</strong>${submission.studentId}</p>
                <p><strong>提交时间：</strong>${submission.submitTime}</p>
                <p><strong>文件名：</strong>${submission.fileName}</p>
            </div>
        `;

        // 根据文件类型显示不同的预览
        if (submission.fileName.toLowerCase().endsWith('.pdf')) {
            previewHtml += `
                <div class="file-preview">
                    <embed src="${submission.fileUrl}" type="application/pdf" width="100%" height="600px">
                </div>
            `;
        } else if (submission.fileName.toLowerCase().match(/\.(doc|docx)$/)) {
            // 对于Word文件，显示提取的文本
            previewHtml += `
                <div class="file-preview">
                    <div class="text-content">
                        <pre>${submission.text}</pre>
                    </div>
                </div>
            `;
        }

        // 添加下载链接
        previewHtml += `
            <div class="download-link">
                <a href="${submission.fileUrl}" download="${submission.fileName}">
                    <button>下载原文件</button>
                </a>
            </div>
        `;

        currentSubmission.innerHTML = previewHtml;

        // 切换到评分区域
        switchSection('grading');
    }

    updateGradedList() {
        const gradedList = document.getElementById('gradedList');
        gradedList.innerHTML = '';

        this.gradedWork.forEach(submission => {
            const div = document.createElement('div');
            div.className = 'submission-item';
            div.innerHTML = `
                <div class="submission-info">
                    <h3>学生：${submission.studentName}</h3>
                    <p>学号：${submission.studentId}</p>
                    <p>提交时间：${submission.submitTime}</p>
                    <p>批改时间：${submission.gradedTime}</p>
                    <p>文件名：${submission.fileName}</p>
                    <p class="grade-info">
                        <strong>分数：</strong>${submission.score}
                        <br>
                        <strong>评语：</strong>${submission.feedback || '无'}
                    </p>
                </div>
                <div class="submission-actions">
                    <button onclick="teacher.viewGradedSubmission(${submission.id})">查看详情</button>
                </div>
            `;
            gradedList.appendChild(div);
        });

        // 更新工作台统计数据
        document.getElementById('gradedCount').textContent = this.gradedWork.length;
    }

    async viewGradedSubmission(submissionId) {
        const submission = this.gradedWork.find(s => s.id === submissionId);
        if (!submission) return;

        const currentSubmission = document.getElementById('currentSubmission');
        currentSubmission.dataset.submissionId = submissionId;

        // 创建预览区域
        let previewHtml = `
            <div class="submission-detail">
                <h3>作业详情</h3>
                <p><strong>学生：</strong>${submission.studentName}</p>
                <p><strong>学号：</strong>${submission.studentId}</p>
                <p><strong>提交时间：</strong>${submission.submitTime}</p>
                <p><strong>批改时间：</strong>${submission.gradedTime}</p>
                <p><strong>文件名：</strong>${submission.fileName}</p>
                <p><strong>分数：</strong>${submission.score}</p>
                <p><strong>评语：</strong>${submission.feedback || '无'}</p>
            </div>
        `;

        // 根据文件类型显示不同的预览
        if (submission.fileName.toLowerCase().endsWith('.pdf')) {
            previewHtml += `
                <div class="file-preview">
                    <embed src="${submission.fileUrl}" type="application/pdf" width="100%" height="600px">
                </div>
            `;
        } else if (submission.fileName.toLowerCase().match(/\.(doc|docx)$/)) {
            previewHtml += `
                <div class="file-preview">
                    <div class="text-content">
                        <pre>${submission.text}</pre>
                    </div>
                </div>
            `;
        }

        // 添加下载链接
        previewHtml += `
            <div class="download-link">
                <a href="${submission.fileUrl}" download="${submission.fileName}">
                    <button>下载原文件</button>
                </a>
            </div>
        `;

        currentSubmission.innerHTML = previewHtml;

        // 切换到评分区域
        switchSection('grading');
    }

    // 添加自动评分方法
    async startAutoGrade() {
        const submissionId = document.getElementById('currentSubmission').dataset.submissionId;
        const standardAnswer = document.getElementById('standardAnswer').value;
        const keywords = document.getElementById('keywords').value.split(',');
        const threshold = parseInt(document.getElementById('similarityThreshold').value);

        if (!submissionId) {
            alert('请先选择要评分的作业');
            return;
        }

        if (!standardAnswer || !keywords.length) {
            alert('请填写标准答案和关键词');
            return;
        }

        const submission = this.submissions.find(s => s.id === parseInt(submissionId));
        if (!submission) return;

        try {
            const token = localStorage.getItem('token');
            if (!token) {
                alert('请先登录');
                return;
            }

            // 先检查 Python 服务是否在运行
            try {
                const healthCheck = await fetch('http://localhost:5000/health');
                if (!healthCheck.ok) {
                    throw new Error('自动评分服务未启动');
                }
            } catch (error) {
                alert('自动评分服务未启动，请确保 Python 服务正在运行');
                return;
            }

            // 调用 Python 服务进行自动评分
            const response = await fetch('http://localhost:5000/api/auto-grade', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    studentAnswer: submission.text || '',
                    standardAnswer,
                    keywords,
                    threshold
                })
            });

            if (!response.ok) {
                throw new Error('自动评分请求失败');
            }

            const result = await response.json();
            
            if (result.success) {
                // 显示评分结果
                document.getElementById('score').value = result.score;
                document.getElementById('feedback').value = result.feedback;
                
                // 切换到手动评分视图以便教师确认或修改
                switchGradingType('manual', document.querySelector('.grading-type-btn'));

                // 添加到历史记录
                const historyItem = {
                    studentName: submission.studentName,
                    studentId: submission.studentId,
                    fileName: submission.fileName,
                    submitTime: submission.submitTime,
                    gradedTime: new Date().toLocaleString(),
                    score: result.score,
                    similarity: result.similarity,
                    matchedKeywords: result.matchedKeywords,
                    feedback: result.feedback
                };
                
                this.autoGradingHistory.unshift(historyItem); // 添加到开头
                this.saveAutoGradingHistory();
                this.updateAutoGradingStats();
                this.updateAutoGradingHistory();
            } else {
                alert(result.message || '自动评分失败');
            }
        } catch (error) {
            console.error('自动评分失败:', error);
            alert(`自动评分失败: ${error.message}`);
        }
    }

    // 加载自动评分历史
    loadAutoGradingHistory() {
        const history = localStorage.getItem('autoGradingHistory');
        this.autoGradingHistory = history ? JSON.parse(history) : [];
        this.updateAutoGradingStats();
        this.updateAutoGradingHistory();
    }

    // 保存自动评分历史
    saveAutoGradingHistory() {
        localStorage.setItem('autoGradingHistory', JSON.stringify(this.autoGradingHistory));
    }

    // 更新自动评分统计
    updateAutoGradingStats() {
        const count = this.autoGradingHistory.length;
        const totalScore = this.autoGradingHistory.reduce((sum, item) => sum + item.score, 0);
        const avgScore = count > 0 ? Math.round(totalScore / count) : 0;

        document.getElementById('autoGradedCount').textContent = count;
        document.getElementById('avgScore').textContent = avgScore;
    }

    // 更新自动评分历史显示
    updateAutoGradingHistory() {
        const historyList = document.getElementById('autoGradingHistory');
        const studentFilter = document.getElementById('studentFilter')?.value.toLowerCase() || '';
        const dateFilter = document.getElementById('dateFilter')?.value || 'all';

        let filteredHistory = this.autoGradingHistory;

        // 应用过滤器
        if (studentFilter) {
            filteredHistory = filteredHistory.filter(item => 
                item.studentName.toLowerCase().includes(studentFilter)
            );
        }

        if (dateFilter !== 'all') {
            const now = new Date();
            const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
            filteredHistory = filteredHistory.filter(item => {
                const itemDate = new Date(item.gradedTime);
                switch(dateFilter) {
                    case 'today':
                        return itemDate >= today;
                    case 'week':
                        const weekAgo = new Date(today - 7 * 24 * 60 * 60 * 1000);
                        return itemDate >= weekAgo;
                    case 'month':
                        const monthAgo = new Date(today.getFullYear(), today.getMonth() - 1, today.getDate());
                        return itemDate >= monthAgo;
                    default:
                        return true;
                }
            });
        }

        // 显示历史记录
        historyList.innerHTML = filteredHistory.map(item => `
            <div class="history-item">
                <div class="history-header">
                    <h4>${item.studentName} (${item.studentId})</h4>
                    <span class="score">${item.score}分</span>
                </div>
                <div class="history-details">
                    <p><strong>提交时间：</strong>${item.submitTime}</p>
                    <p><strong>评分时间：</strong>${item.gradedTime}</p>
                    <p><strong>文件名：</strong>${item.fileName}</p>
                    <p><strong>相似度：</strong>${item.similarity.toFixed(2)}%</p>
                    <p><strong>关键词匹配：</strong>${item.matchedKeywords.join(', ')}</p>
                    <div class="feedback">
                        <strong>评语：</strong>
                        <pre>${item.feedback}</pre>
                    </div>
                </div>
            </div>
        `).join('');
    }
}

const teacher = new TeacherSystem();
teacher.fetchSubmissions(); 

function switchGradingType(type, button) {
    const manualGrading = document.getElementById('manualGrading');
    const autoGrading = document.getElementById('autoGrading');
    const buttons = document.querySelectorAll('.grading-type-btn');
    
    buttons.forEach(btn => btn.classList.remove('active'));
    button.classList.add('active');
    
    if (type === 'manual') {
        manualGrading.style.display = 'block';
        autoGrading.style.display = 'none';
    } else {
        manualGrading.style.display = 'none';
        autoGrading.style.display = 'block';
    }
}

// 更新相似度阈值显示
document.getElementById('similarityThreshold')?.addEventListener('input', (e) => {
    document.getElementById('thresholdValue').textContent = `${e.target.value}%`;
});

function startAutoGrading() {
    const standardAnswer = document.getElementById('standardAnswer').value;
    const keywords = document.getElementById('keywords').value.split(',');
    const threshold = parseInt(document.getElementById('similarityThreshold').value);

    if (!standardAnswer || !keywords.length) {
        alert('请填写标准答案和关键词');
        return;
    }

    // 假设我有一选中的作业
    const submissionId = document.getElementById('currentSubmission').dataset.submissionId;
    if (!submissionId) {
        alert('请先选择要评分的作业');
        return;
    }

    teacher.autoGrade(submissionId);
} 

function switchSection(sectionId) {
    // 移除所有 section 的 active 类
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    
    // 移除所有导航项的 active 类
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // 添加 active 类到选中的 section
    document.getElementById(sectionId).classList.add('active');
    
    // 找到对应的导航项并添加 active 类
    const navItem = document.querySelector(`.nav-item[onclick="switchSection('${sectionId}')"]`);
    if (navItem) {
        navItem.classList.add('active');
    }
} 

// 添加事件监听器
document.getElementById('studentFilter')?.addEventListener('input', () => {
    teacher.updateAutoGradingHistory();
});

document.getElementById('dateFilter')?.addEventListener('change', () => {
    teacher.updateAutoGradingHistory();
}); 