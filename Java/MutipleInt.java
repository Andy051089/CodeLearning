/**
 * 數組:一組能夠存處相同數據類型的數據集合
 * 數組一定要長度
 * 數組中的每個數據稱為元素
 * 數組元素從0開始
 * 數組中的位置稱為下標
 */
package hello;

public class MutipleInt {
	public static void main(String[] args) {
		//第一種
		int[]nums = new int[5];
		/**
		nums[0] = 1;
		nums[1] = 2;
		nums[2] = 3;
		nums[3] = 4;
		nums[4] = 5;
		*/
		for(int i = 0;i<5;i++) {
			nums[i] = i+1;
		}
		//第二種
		int[]nums2; //先聲明
		nums2 = new int[5];
		//第三種
		int[]nums3 = new int[] {1,2,3,4,5};
		//第四種
		int[]nums4 =  {1,2,3,4,5};
		System.out.println(nums.length);
	}
}
